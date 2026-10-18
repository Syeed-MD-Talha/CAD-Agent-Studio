"""Measurements of completed BREP geometry, independent of script variables.

All dimensions are mm. Unsupported or ambiguous checks remain Not checked.
"""

import math
import re
from itertools import combinations

KINDS = {
    "extent": "Overall dimension",
    "hole_diameter": "Cylindrical bore diameter",
    "hole_count": "Cylindrical bore count",
    "solid_count": "Solid count",
    "object_count": "Displayed object count",
    "min_clearance": "Minimum clearance",
    "max_overlap_volume": "Maximum overlap volume",
    "wall_thickness": "Minimum wall thickness",
}


def normalize_requirements(requirements):
    if not isinstance(requirements, list) or len(requirements) > 100:
        raise ValueError("Requirements must be a list of at most 100 checks.")
    normalized = []
    for raw in requirements:
        if not isinstance(raw, dict) or raw.get("kind") not in KINDS:
            raise ValueError("Unknown requirement type.")
        item = {
            key: str(raw.get(key, ""))[:300]
            for key in ("label", "target", "other", "source")
        }
        item["kind"] = raw["kind"]
        item["axis"] = str(raw.get("axis", "X")).upper()
        if item["axis"] not in ("X", "Y", "Z"):
            raise ValueError("Dimension axis must be X, Y, or Z.")
        for key, default in (("expected", None), ("tolerance", 0.01)):
            value = float(raw.get(key, default))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative.")
            item[key] = value
        if item["kind"] in ("solid_count", "object_count", "hole_count"):
            if not item["expected"].is_integer():
                raise ValueError("Counts must be whole numbers.")
            item["tolerance"] = 0.0
        normalized.append(item)
    return normalized


def row(label, status, measured="", expected="", detail=""):
    return dict(
        label=label, status=status, measured=measured, expected=expected, detail=detail
    )


def _intersection_volume(first, second):
    """build123d can return None, an empty shape, or contact faces for a common."""
    common = first & second
    if common is None:
        return 0.0
    from build123d import Shape

    if not isinstance(common, Shape):
        raise TypeError(
            "Unsupported intersection result; overlap could not be measured."
        )
    wrapped = getattr(common, "_wrapped", None)
    if wrapped is None or wrapped.IsNull():
        return 0.0
    # Face/edge contact has no interfering solid volume.
    volume = math.fsum(abs(solid.volume) for solid in common.solids())
    if not math.isfinite(volume):
        raise ValueError("The intersection did not produce a finite volume.")
    return volume


def _requirement_target(check, names):
    """Resolve only an exact, unambiguous displayed name in the requirement.

    Explicit targets (including 'all') always take precedence. Expose any
    inferred scope in the report instead of silently changing the saved check.
    """
    if check["target"]:
        return check["target"], ""
    if check["kind"] in ("object_count", "min_clearance", "max_overlap_volume"):
        return "", ""
    text = check["label"] + " " + check["source"]
    matches = [
        name
        for name in names
        if name.lower() not in ("all", "result", "model", "part", "shape")
        and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE)
    ]
    if len(matches) == 1:
        return (
            matches[0],
            f"Scope: displayed object '{matches[0]}' (matched its exact name in the requirement; use Checks to set an explicit target). ",
        )
    return "", "Scope: all displayed objects. "


def _bores(shape):
    """Deduplicate inward cylindrical faces by radius and their infinite axis.

    This measures cylindrical bores only; threads, conical and irregular holes
    require other inspection. Counterbore stages count as separate bores.
    """
    import build123d as b
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    bores = {}
    solids = list(shape.solids())
    faces = (
        [(i, face) for i, solid in enumerate(solids) for face in solid.faces()]
        if solids
        else [(0, face) for face in shape.faces()]
    )
    for solid_index, face in faces:
        if face.geom_type != b.GeomType.CYLINDER:
            continue
        cylinder = BRepAdaptor_Surface(face.wrapped).Cylinder()
        origin = b.Vector(cylinder.Location())
        direction = b.Vector(cylinder.Axis().Direction())
        point = face.position_at(0.5, 0.5)
        offset = point - origin
        radial = offset - direction * offset.dot(direction)
        if face.normal_at(0.5, 0.5).dot(radial) >= 0:
            continue  # external cylindrical surfaces are not holes
        components = tuple(direction)
        if next((x for x in components if abs(x) > 1e-7), 1) < 0:
            direction = -direction
        closest = origin - direction * origin.dot(direction)
        key = (
            solid_index,
            *(
                round(x, 5)
                for x in (*tuple(direction), *tuple(closest), cylinder.Radius())
            ),
        )
        bores[key] = 2 * cylinder.Radius()
    return list(bores.values())


def inspect_geometry(objects, requirements=None, assembly=True):
    import build123d as b

    requirements = normalize_requirements(requirements or [])
    shapes = {name: shape for name, shape in objects.items()}
    rows = []
    valid = bool(shapes) and all(
        s.is_valid and bool(s.vertices()) for s in shapes.values()
    )
    rows.append(
        row(
            "Nonempty valid geometry",
            "Passed" if valid else "Failed",
            str(len(shapes)) + " objects",
        )
    )
    if not valid:
        return {"rows": rows, "ok": False, "requirements": requirements}
    whole = b.Compound(children=list(shapes.values()))
    dims = whole.bounding_box().size
    rows.append(
        row(
            "Measured overall size (X × Y × Z)",
            "Measured",
            " × ".join(f"{v:.4g}" for v in dims) + " mm",
        )
    )
    rows.append(row("Solids", "Measured", str(len(whole.solids()))))

    multi_solid = {
        name: len(shape.solids())
        for name, shape in shapes.items()
        if len(shape.solids()) > 1
    }
    if multi_solid:
        rows.append(
            row(
                "Multi-solid displayed objects",
                "Not checked",
                "; ".join(
                    f"{name}: {count} solids" for name, count in multi_solid.items()
                ),
                detail="A displayed object is not necessarily one continuous part. Confirm these solids are intentional; use a named solid-count check to require one continuous piece.",
            )
        )

    bore_cache = {}
    pair_cache = {}

    def select(target):
        if not target or target == "all":
            return whole
        if target not in shapes:
            raise ValueError(f"Displayed object '{target}' was not found.")
        return shapes[target]

    def pair_values(first, second):
        key = tuple(sorted((first, second)))
        if first == second:
            raise ValueError("Choose two different objects.")
        if first not in shapes or second not in shapes:
            raise ValueError(
                "Choose two named displayed objects for the assembly check."
            )
        if key not in pair_cache:
            a, c = select(first), select(second)
            pair_cache[key] = (a.distance_to(c), _intersection_volume(a, c))
        return pair_cache[key]

    for check in requirements:
        label = check["label"] or KINDS[check["kind"]]
        detail = (
            "Source: " + check["source"]
            if check["source"]
            else "Check supplied by the assistant/user; confirm against the request."
        )
        expected = check["expected"]
        tolerance = check["tolerance"]
        target, scope_note = _requirement_target(check, shapes)
        detail = scope_note + detail
        if check["target"]:
            detail = f"Scope: {check['target']}. " + detail
        kind = check["kind"]
        unit = (
            ""
            if kind.endswith("count")
            else (" mm³" if kind == "max_overlap_volume" else " mm")
        )
        expected_text = f"{expected:g} ± {tolerance:g}{unit}"
        try:
            shape = select(target)
            if kind == "extent":
                actual = getattr(shape.bounding_box().size, check["axis"])
            elif kind == "solid_count":
                actual = len(shape.solids())
            elif kind == "object_count":
                actual = len(shapes) if not target or target == "all" else 1
            elif kind in ("hole_count", "hole_diameter"):
                if target not in bore_cache:
                    bore_cache[target] = _bores(shape)
                diameters = bore_cache[target]
                if kind == "hole_count":
                    actual = len(diameters)
                else:
                    if not diameters:
                        raise ValueError(
                            "No inward cylindrical bore face found; threaded, conical or irregular holes need other checks."
                        )
                    passed = all(
                        abs(value - expected) <= tolerance for value in diameters
                    )
                    rows.append(
                        row(
                            label,
                            "Passed" if passed else "Failed",
                            ", ".join(f"{value:.6g}" for value in diameters) + " mm",
                            expected_text,
                            detail
                            + " All cylindrical bores in this target; position, depth and threads are not checked. Different bore sizes need separate named targets.",
                        )
                    )
                    continue
            elif kind in ("min_clearance", "max_overlap_volume"):
                if not target or not check["other"]:
                    raise ValueError(
                        "Choose two named displayed objects for the assembly check."
                    )
                distance, volume = pair_values(target, check["other"])
                actual = distance if kind == "min_clearance" else volume
            else:
                raise ValueError(
                    "Automatic minimum wall thickness is not implemented; inspect a section or add a specific geometric dimension check."
                )
            passed = (
                actual + tolerance >= expected
                if kind == "min_clearance"
                else (
                    actual <= expected + tolerance
                    if kind == "max_overlap_volume"
                    else abs(actual - expected) <= tolerance
                )
            )
            rows.append(
                row(
                    label,
                    "Passed" if passed else "Failed",
                    f"{actual:.6g}{unit}",
                    expected_text,
                    detail,
                )
            )
        except Exception as error:
            rows.append(row(label, "Not checked", "", expected_text, str(error)))

    solids = list(whole.solids())
    if assembly and 1 < len(solids) <= 15:
        overlaps, nearest, errors = [], None, []
        for i, j in combinations(range(len(solids)), 2):
            a, c = solids[i], solids[j]
            try:
                distance = a.distance_to(c)
                if not math.isfinite(distance) or distance < 0:
                    raise ValueError("Invalid distance result.")
                nearest = distance if nearest is None else min(nearest, distance)
                if distance < 1e-6:
                    volume = _intersection_volume(a, c)
                    if volume > 1e-6:
                        overlaps.append(f"solid {i + 1}/{j + 1}: {volume:.5g} mm³")
            except Exception as error:
                errors.append(f"solid {i + 1}/{j + 1}: {type(error).__name__}: {error}")
        rows.append(
            row(
                "Assembly interference",
                "Failed" if overlaps else ("Not checked" if errors else "Passed"),
                "; ".join(overlaps)
                or ("Incomplete inspection" if errors else "No volumetric overlap"),
                "≤ 0.000001 mm³",
                "Intentional overlaps need review; solids are numbered across displayed objects. "
                + "; ".join(errors),
            )
        )
        rows.append(
            row(
                "Nearest solid clearance",
                "Not checked" if errors else "Measured",
                f"{nearest:.6g} mm" if nearest is not None else "",
                detail=(
                    "Some pair measurements were unavailable. " + "; ".join(errors)
                    if errors
                    else ""
                ),
            )
        )
    elif assembly and len(solids) > 15:
        rows.append(
            row(
                "Assembly interference",
                "Not checked",
                detail="More than 15 solids; use explicit named-pair checks to bound runtime.",
            )
        )
    elif assembly:
        rows.append(
            row(
                "Assembly interference",
                "Not checked",
                detail="Requires at least two solids.",
            )
        )
    if not requirements:
        rows.append(
            row(
                "Requested dimensions and features",
                "Not checked",
                detail="Add checks or ask the assistant to record the requirements from your prompt.",
            )
        )
    rows.append(
        row(
            "Minimum wall thickness / strength / manufacturing",
            "Not checked",
            detail="No automatic thickness scan or structural/manufacturing analysis is performed.",
        )
    )
    rows.append(
        row(
            "AI visual review",
            "Not checked",
            detail="Enable Vision to capture views for the AI; an AI assessment is separate from measured checks.",
        )
    )
    return {
        "rows": rows,
        "ok": not any(r["status"] == "Failed" for r in rows),
        "requirements": requirements,
    }


def report_text(report):
    return "\n".join(
        f"{r['status']}: {r['label']} | measured: {r.get('measured') or '—'} | expected: {r.get('expected') or '—'}"
        + (" | " + r["detail"] if r.get("detail") else "")
        for r in report.get("rows", [])
    )


def inspect_stl(path):
    """Check exported triangle edges and connected shells, in the child only."""
    import struct
    from collections import Counter
    from pathlib import Path

    if Path(path).stat().st_size > 256 * 1024 * 1024:
        return [
            row(
                "STL watertightness",
                "Not checked",
                detail="Mesh exceeds the 256 MB inspection budget.",
            )
        ]
    data = Path(path).read_bytes()
    binary_count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + binary_count * 50:
        if binary_count > 500000:
            return [
                row(
                    "STL watertightness",
                    "Not checked",
                    detail="Mesh exceeds the 500,000 triangle inspection budget.",
                )
            ]
        triangles = (
            struct.unpack_from("<9f", data, 84 + i * 50 + 12)
            for i in range(binary_count)
        )
    else:
        import re

        vertices = re.findall(
            rb"vertex\s+([-+\deE.]+)\s+([-+\deE.]+)\s+([-+\deE.]+)", data
        )
        if len(vertices) % 3:
            return [
                row(
                    "STL triangle data",
                    "Failed",
                    detail="Incomplete triangle in exported STL.",
                )
            ]
        if len(vertices) > 1500000:
            return [
                row(
                    "STL watertightness",
                    "Not checked",
                    detail="Mesh exceeds the 500,000 triangle inspection budget.",
                )
            ]
        triangles = (
            tuple(float(v) for vertex in vertices[i : i + 3] for v in vertex)
            for i in range(0, len(vertices), 3)
        )
    edges, directed = Counter(), Counter()
    parent = {}
    degenerate = 0

    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    count = 0
    for coords in triangles:
        count += 1
        points = [tuple(round(v, 6) for v in coords[i : i + 3]) for i in (0, 3, 6)]
        if any(not math.isfinite(v) for v in coords):
            return [
                row(
                    "STL triangle data",
                    "Failed",
                    detail="Non-finite triangle coordinates.",
                )
            ]
        a, c, d = points
        u, v = [c[i] - a[i] for i in range(3)], [d[i] - a[i] for i in range(3)]
        cross = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        if len(set(points)) < 3 or sum(x * x for x in cross) <= 1e-24:
            degenerate += 1
        for first, second in ((a, c), (c, d), (d, a)):
            key = tuple(sorted((first, second)))
            edges[key] += 1
            directed[(first, second)] += 1
            parent[root(second)] = root(first)
    boundary = sum(n == 1 for n in edges.values())
    nonmanifold = sum(n != 2 for n in edges.values())
    coherent = all(directed[(a, c)] == directed[(c, a)] for a, c in edges)
    passed = count > 0 and nonmanifold == 0 and degenerate == 0 and coherent
    return [
        row(
            "STL watertight oriented mesh",
            "Passed" if passed else "Failed",
            f"{count} triangles; {boundary} boundary edges; {nonmanifold} nonmanifold edges; {degenerate} degenerate triangles",
            "Closed, consistently oriented triangle mesh",
            "Vertices are matched within 0.000001 mm; this is not a self-intersection test.",
        ),
        row(
            "STL connected components",
            "Measured",
            str(len({root(key) for key in parent})),
            detail="Mesh connectivity can differ from CAD solid count when components touch or a solid has internal cavities.",
        ),
    ]
