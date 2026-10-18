"""Exercise real build123d geometry and the errors seen in generated scripts."""

from pathlib import Path

import pytest

from cq_editor.ai.cad_agent import isolated_validate, read_example


def test_examples_return_source_and_reject_arbitrary_paths():
    assert "09_threaded_bolt_and_nut" in read_example()
    source = read_example("09_threaded_bolt_and_nut.py")
    assert "```python" in source
    assert "is_frenet=True" in source
    for name in ("../README", "C:/Windows/system.ini", "missing", 42):
        assert read_example(name).startswith("Error:")


@pytest.mark.parametrize(
    "expression, guidance",
    [
        ("Rectangle(2, 3).located(Plane.XZ)", "plane.location"),
        ("Rectangle(2, 3).extrude(amount=5)", "global extrude(sketch, amount=height)"),
    ],
)
def test_actual_api_failures_report_source_and_repair_guidance(expression, guidance):
    code = "from build123d import *\nresult = " + expression + "\nshow_object(result)"
    result = isolated_validate(code)
    assert result.startswith("Traceback:")
    assert "script line 2" in result
    assert "Failing code: result = " + expression in result
    assert guidance in result


@pytest.mark.parametrize("expression", ["None", "Compound()"])
def test_empty_geometry_is_never_reported_as_success(expression):
    result = isolated_validate(
        "from build123d import *\nshow_object(" + expression + ")"
    )
    assert result.startswith("Error:")
    assert "empty, invalid, or non-build123d geometry" in result


def test_real_threaded_bolt_and_nut_are_valid_separate_solids(tmp_path):
    # Run in the same crash-isolated process used before applying AI code.
    source = (
        Path(__file__).resolve().parents[1] / "examples" / "09_threaded_bolt_and_nut.py"
    ).read_text()
    checks = """
assert bolt.is_valid and nut.is_valid
assert len(bolt.solids()) == len(nut.solids()) == 1
assert len(result.solids()) == 2
assert abs(bolt.bounding_box().size.Z - (shaft_length + head_height)) < 0.001
assert abs(nut.bounding_box().size.Z - nut_height) < 0.001
# The shaft has real helical faces, rather than only a cylinder surface.
assert len(shaft.faces()) > 5
export_step(result, "threaded.step")
"""
    verdict = isolated_validate(source + checks, workdir=tmp_path)
    assert verdict.startswith("OK:"), verdict
    assert "2 object(s)" in verdict
    assert (tmp_path / "threaded.step").stat().st_size > 10000
