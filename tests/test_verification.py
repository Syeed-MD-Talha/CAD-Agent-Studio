"""Real measurements, export round trips, and a responsive cancellable CAD job."""

import time
import pytest
from PyQt5.QtCore import QTimer

from cq_editor.ai.verification import (
    inspect_geometry,
    inspect_stl,
    normalize_requirements,
)
from cq_editor.widgets.cad_job import run_cad_job


def requirement(kind, expected, **kwargs):
    return dict(kind=kind, expected=expected, **kwargs)


def test_actual_geometry_catches_wrong_dimensions_and_mixed_bores():
    from build123d import Box, Cylinder, Pos

    plate = (
        Box(40, 20, 4)
        - Pos(-8, 0, 0) * Cylinder(2, 10)
        - Pos(8, 0, 0) * Cylinder(3, 10)
    )
    report = inspect_geometry(
        {"plate": plate},
        [
            requirement("extent", 40, axis="X", label="Length"),
            requirement("extent", 5, axis="Z", label="Thickness"),
            requirement("hole_count", 2, label="Holes"),
            requirement("hole_diameter", 4, label="All bores 4 mm"),
            requirement("wall_thickness", 2, label="Wall thickness"),
        ],
    )
    status = {r["label"]: r["status"] for r in report["rows"]}
    assert status["Length"] == status["Holes"] == "Passed"
    assert status["Thickness"] == status["All bores 4 mm"] == "Failed"
    assert status["Wall thickness"] == "Not checked"
    assert not report["ok"]


def test_assembly_overlap_and_clearance_are_measured():
    from build123d import Box, Pos

    a = Box(10, 10, 10)
    overlap = inspect_geometry({"a": a, "b": Pos(8, 0, 0) * a})
    assert (
        next(r for r in overlap["rows"] if r["label"] == "Assembly interference")[
            "status"
        ]
        == "Failed"
    )
    separated = inspect_geometry(
        {"a": a, "b": Pos(13, 0, 0) * a},
        [
            requirement("min_clearance", 3, target="a", other="b", label="Gap"),
            requirement("solid_count", 2, label="Parts"),
        ],
    )
    assert separated["ok"]
    assert (
        next(r for r in separated["rows"] if r["label"] == "Gap")["measured"] == "3 mm"
    )


def test_none_common_is_zero_volume_for_contact_and_explicit_pair_checks(monkeypatch):
    from build123d import Box, Part, Solid, Pos
    from cq_editor.ai import verification

    a, b = Box(10, 10, 10), Pos(10, 0, 0) * Box(10, 10, 10)
    # Reproduce build123d's None result deterministically on both entry paths.
    monkeypatch.setattr(Part, "__and__", lambda self, other: None)
    monkeypatch.setattr(Solid, "__and__", lambda self, other: None)
    assert verification._intersection_volume(a, b) == 0
    report = inspect_geometry(
        {"a": a, "b": b},
        [
            requirement(
                "max_overlap_volume", 0, target="a", other="b", label="Pair overlap"
            ),
            requirement(
                "min_clearance", 0, target="a", other="b", label="Pair clearance"
            ),
        ],
    )
    for label in ("Pair overlap", "Pair clearance", "Assembly interference"):
        assert (
            next(r for r in report["rows"] if r["label"] == label)["status"] == "Passed"
        )
    assert report["ok"]


@pytest.mark.parametrize("offset", [(10, 0, 0), (10, 10, 0), (10, 10, 10), (12, 0, 0)])
def test_real_face_edge_vertex_contact_and_disjoint_shapes_have_no_overlap(offset):
    from build123d import Box, Pos

    report = inspect_geometry(
        {"a": Box(10, 10, 10), "b": Pos(*offset) * Box(10, 10, 10)}
    )
    assert (
        next(r for r in report["rows"] if r["label"] == "Assembly interference")[
            "status"
        ]
        == "Passed"
    )


def test_assembly_checker_errors_remain_not_checked_instead_of_aborting(monkeypatch):
    from build123d import Box, Pos
    from cq_editor.ai import verification

    def unavailable(*args):
        raise RuntimeError("injected kernel measurement failure")

    monkeypatch.setattr(verification, "_intersection_volume", unavailable)
    report = inspect_geometry(
        {"a": Box(10, 10, 10), "b": Pos(10, 0, 0) * Box(10, 10, 10)}
    )
    interference = next(
        r for r in report["rows"] if r["label"] == "Assembly interference"
    )
    assert interference["status"] == "Not checked"
    assert "injected kernel measurement failure" in interference["detail"]
    assert report["ok"]


def test_jar_bore_is_scoped_to_named_jar_and_explicit_all_counts_both_cavities():
    from build123d import Align, Cylinder, Pos

    jar = Cylinder(10, 15, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Pos(
        0, 0, 2
    ) * Cylinder(8, 15, align=(Align.CENTER, Align.CENTER, Align.MIN))
    cap = Pos(30, 0, 0) * (
        Cylinder(12, 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
        - Pos(0, 0, 2) * Cylinder(10, 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
    )
    report = inspect_geometry(
        {"jar": jar, "knurled cap": cap},
        [
            requirement(
                "hole_count",
                1,
                label="Jar has a hollow bore (open container)",
                source="open-top jar in reference image",
            ),
            requirement("hole_count", 2, target="all", label="Combined cavities"),
        ],
    )
    bore = next(r for r in report["rows"] if r["label"].startswith("Jar has"))
    assert bore["status"] == "Passed" and bore["measured"] == "1"
    assert "Scope: displayed object 'jar'" in bore["detail"]
    assert (
        next(r for r in report["rows"] if r["label"] == "Combined cavities")["status"]
        == "Passed"
    )
    assert (
        report["requirements"][0]["target"] == ""
    )  # no saved constraints are rewritten


def test_verifier_exception_does_not_discard_completed_geometry(monkeypatch, tmp_path):
    from cq_editor.ai import verification
    from cq_editor.ai.cad_runtime import run_job

    def unavailable(*args, **kwargs):
        raise AttributeError("injected verifier bug")

    monkeypatch.setattr(verification, "inspect_geometry", unavailable)
    result = run_job(
        dict(
            code="from build123d import *\nshow_object(Box(10,10,4))",
            requirements=[requirement("extent", 10, label="Width")],
        ),
        tmp_path,
    )
    assert result["ok"] and len(result["objects"]) == 1
    assert "injected verifier bug" in result["report"]["verification_error"]
    assert (
        next(r for r in result["report"]["rows"] if r["label"] == "Width")["status"]
        == "Not checked"
    )


def test_displayed_object_with_multiple_solids_is_flagged_for_review():
    from build123d import Box, Compound, Pos

    report = inspect_geometry(
        {"jar": Compound(children=[Box(10, 10, 4), Pos(20, 0, 0) * Box(5, 5, 5)])}
    )
    result = next(
        row for row in report["rows"] if row["label"] == "Multi-solid displayed objects"
    )
    assert result["status"] == "Not checked"
    assert result["measured"] == "jar: 2 solids"


def test_unknown_part_and_unprovided_specs_remain_not_checked():
    from build123d import Box

    report = inspect_geometry(
        {"box": Box(10, 10, 10)},
        [requirement("extent", 10, target="missing", label="Missing part")],
    )
    assert (
        next(r for r in report["rows"] if r["label"] == "Missing part")["status"]
        == "Not checked"
    )
    report = inspect_geometry({"box": Box(10, 10, 10)})
    assert any(
        r["label"] == "Requested dimensions and features"
        and r["status"] == "Not checked"
        for r in report["rows"]
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_check_numbers_are_rejected(value):
    with pytest.raises(ValueError):
        normalize_requirements([requirement("extent", value)])


def test_real_child_build_returns_geometry_and_preserves_names(qapp, tmp_path):
    source = "from build123d import *\nplate = Box(30,20,4) - Cylinder(2,10)\nshow_object(plate)\nprint('RESULT fake result')"
    job = run_cad_job(
        dict(
            code=source,
            workdir=str(tmp_path),
            requirements=[requirement("extent", 30, axis="X")],
        )
    )
    assert job["ok"], job
    assert job["objects"][0]["name"] == "plate"
    assert job["objects"][0]["shape"].is_valid
    assert job["report"]["ok"]


@pytest.mark.parametrize("fmt", ["step", "stl", "brep"])
def test_exports_are_reopened_and_checked(qapp, tmp_path, fmt):
    from build123d import Box, Cylinder

    path = tmp_path / ("plate." + fmt)
    job = run_cad_job(
        dict(assembly=False, export=dict(format=fmt, filename=str(path))),
        shapes={"plate": Box(30, 20, 4) - Cylinder(2, 10)},
    )
    assert job["ok"], job
    assert job["report"]["ok"], job["report"]
    assert path.stat().st_size > 100
    assert any(
        "reopened" in r["label"] and r["status"] == "Passed"
        for r in job["report"]["rows"]
    )
    if fmt == "stl":
        assert (
            next(
                r
                for r in job["report"]["rows"]
                if r["label"] == "STL watertight oriented mesh"
            )["status"]
            == "Passed"
        )


def test_open_mesh_fails_watertightness(tmp_path):
    path = tmp_path / "open.stl"
    path.write_text(
        "solid open\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid open"
    )
    assert inspect_stl(path)[0]["status"] == "Failed"


def test_gui_timers_keep_ticking_and_stop_kills_infinite_code(qapp, tmp_path):
    beats = []
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: beats.append(time.monotonic()))
    timer.start()
    started = time.monotonic()
    ready = tmp_path / "ready.txt"
    source = f"from pathlib import Path\nPath({str(ready)!r}).write_text('running')\nwhile True: pass"
    job = run_cad_job(dict(code=source), timeout=30, cancelled=ready.exists)
    timer.stop()
    assert len(beats) >= 10
    assert ready.exists()
    assert time.monotonic() - started < 20
    assert not job["ok"] and "Cancelled" in job["error"]


def test_child_timeout_is_bounded(qapp):
    started = time.monotonic()
    job = run_cad_job(dict(code="while True: pass"), timeout=0.3)
    assert not job["ok"] and job["error"].startswith("TIMEOUT:")
    assert time.monotonic() - started < 4


def test_script_exception_returns_source_line_without_changing_geometry(qapp):
    job = run_cad_job(
        dict(
            code="from build123d import *\nresult = Box(10,10,10)\nraise ValueError('intentional failure')\nshow_object(result)"
        )
    )
    assert not job["ok"]
    assert job["line"] == 3
    assert "intentional failure" in job["error"]


def test_native_worker_crash_is_contained(qapp):
    job = run_cad_job(dict(code="import os\nos._exit(23)"))
    assert not job["ok"] and job["error"].startswith("CRASH:")


def test_assembly_labels_colors_and_nested_placement_survive_transfer(qapp):
    source = """from build123d import *
a = Box(4, 4, 4)
a.label = "red block"
a.color = Color("red")
b = Pos(12, 0, 0) * Box(4, 4, 4)
b.label = "other block"
result = Pos(20, 10, 0) * Compound(children=[a, b])
show_object(result, name="assembly")
"""
    job = run_cad_job(dict(code=source))
    assert job["ok"], job
    shape = job["objects"][0]["shape"]
    assert [child.label for child in shape.children] == ["red block", "other block"]
    assert tuple(shape.children[0].color) == (1, 0, 0, 1)
    assert shape.bounding_box().min.X == pytest.approx(18)
    assert shape.bounding_box().min.Y == pytest.approx(8)
    assert shape.bounding_box().size.X == pytest.approx(16)


def test_manual_render_retains_topods_lists_and_console_shape_variables(qapp):
    source = """from build123d import *
from OCP.AIS import AIS_Shape
plate = Box(10, 10, 4)
other = Pos(20,0,0) * Box(5,5,5)
show_object(plate.wrapped)
show_object([plate, other])
ais = AIS_Shape(other.wrapped)
show_object(ais)
"""
    job = run_cad_job(dict(code=source, auto_show=True))
    assert job["ok"], job
    assert len(job["objects"]) == 3
    assert {item["name"] for item in job["extra_shapes"]} == {"plate", "other"}
    assert all(item["shape"].is_valid for item in job["extra_shapes"])


def test_manual_utility_script_can_finish_without_geometry(qapp):
    job = run_cad_job(
        dict(code="print('utility completed')\nanswer=42", auto_show=True)
    )
    assert job["ok"] and not job["objects"]
    assert job["locals"]["answer"] == 42
    assert "utility completed" in job["log"]
    assert job["report"]["rows"][0]["status"] == "Not checked"
