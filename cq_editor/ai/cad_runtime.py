"""Crash-isolated CAD worker. File protocol avoids script stdout corruption."""

import json
import math
import random
import sys
import traceback
from pathlib import Path


def run_job(request, folder):
    import build123d as b
    from .verification import inspect_geometry, inspect_stl, row

    folder = Path(folder)
    shown = {}
    options = {}
    namespace = {}
    random.seed(59798267586177)

    def rand_color(alpha=0.0, cfloat=False):
        values = tuple(random.randrange(10, 100) for _ in range(3))
        return (
            (*tuple(v / 255 for v in values), alpha)
            if cfloat
            else {"alpha": alpha, "color": values}
        )

    def coerce_shape(obj):
        if isinstance(obj, b.BuildPart):
            return obj.part
        if isinstance(obj, b.BuildSketch):
            return obj.sketch
        if isinstance(obj, b.BuildLine):
            return obj.line
        if request.get("auto_show"):
            from OCP.TopoDS import TopoDS_Shape
            from OCP.AIS import AIS_Shape

            if isinstance(obj, TopoDS_Shape):
                return b.Shape.cast(obj)
            if isinstance(obj, AIS_Shape):
                return b.Shape.cast(obj.Shape())
            if isinstance(obj, (list, tuple)):
                return b.Compound(children=[coerce_shape(item) for item in obj])
        return obj

    if request.get("input_shapes") is not None:
        shown = {
            item["name"]: b.import_brep(item["path"])
            for item in request["input_shapes"]
        }
    else:

        def show_object(obj, name=None, options=None, **kwargs):
            original = obj
            obj = coerce_shape(obj)
            if not isinstance(obj, b.Shape) or not obj.is_valid or not obj.vertices():
                raise ValueError(
                    "show_object received empty, invalid, or non-build123d geometry"
                )
            if not name:
                import inspect

                caller = inspect.currentframe().f_back
                name = next(
                    (
                        key
                        for key, value in caller.f_locals.items()
                        if value is original and not key.startswith("_")
                    ),
                    None,
                )
            name = str(name or getattr(obj, "label", "") or str(id(original)))
            shown[name] = obj
            options_dict = options or {}
            # Only JSON-compatible display options cross the process boundary.
            try:
                json.dumps(options_dict, allow_nan=False)
            except (ValueError, TypeError):
                options_dict = {}
            display_options[name] = options_dict

        display_options = options
        namespace = dict(
            b3d=b,
            build123d=b,
            show_object=show_object,
            debug=lambda obj, name=None: show_object(
                obj, name, {"color": "red", "alpha": 0.2}
            ),
            log=lambda value: print(value),
            rand_color=rand_color,
            __name__="__cq_main__",
            __file__=request.get("filename") or "<cad-worker>",
        )
        source = request.get("code", "")
        exec(compile(source, "<cad-worker>", "exec"), namespace, namespace)
        if not shown and request.get("auto_show"):
            shown = {
                name: value
                for name, value in namespace.items()
                if isinstance(value, b.Shape)
            }
    if not shown and not request.get("auto_show"):
        raise ValueError("No geometry was shown. Add show_object(result).")
    if shown:
        try:
            report = inspect_geometry(
                shown,
                request.get("requirements"),
                assembly=request.get("assembly", True),
            )
        except Exception as error:
            # A checker exception must not be misreported as failed user code
            # or discard completed geometry. Keep every requested check visible.
            detail = f"Verifier error: {type(error).__name__}: {error}. Geometry was built; these measurements were not completed."
            rows = [
                row("Script execution", "Passed", str(len(shown)) + " objects"),
                row("Design verification", "Not checked", detail=detail),
            ]
            rows.extend(
                row(
                    check.get("label") or check.get("kind", "Requirement"),
                    "Not checked",
                    expected=str(check.get("expected", "")),
                    detail=detail,
                )
                for check in request.get("requirements", [])
                if isinstance(check, dict)
            )
            report = dict(
                ok=True,
                rows=rows,
                requirements=request.get("requirements", []),
                verification_error=detail,
            )
    else:
        report = dict(
            ok=True,
            rows=[
                row(
                    "Geometry",
                    "Not checked",
                    detail="Script completed without producing geometry.",
                )
            ],
            requirements=[],
        )
    manifest = []
    tree_index = 0

    def serialize_tree(shape):
        nonlocal tree_index
        tree_index += 1
        node = dict(label=getattr(shape, "label", ""))
        if shape.color is not None:
            node["color"] = tuple(shape.color)
        children = list(getattr(shape, "children", ()))
        if children:
            node["location"] = (
                tuple(shape.location.position),
                tuple(shape.location.orientation),
            )
            node["children"] = [serialize_tree(child) for child in children]
        else:
            path = folder / f"tree-{tree_index}.brep"
            b.export_brep(shape, path)
            node["path"] = str(path)
        return node

    for i, (name, shape) in enumerate(shown.items()):
        if request.get("prepare_display", True) and shape.solids():
            shape.mesh(0.1, 0.2)
        path = folder / f"object-{i}.brep"
        if not b.export_brep(shape, path):
            raise ValueError("Could not transfer completed geometry")
        item = dict(name=name, path=str(path), options=options.get(name, {}))
        if getattr(shape, "children", ()):
            item["tree"] = serialize_tree(shape)
        manifest.append(item)
    if request.get("export"):
        fmt = request["export"]["format"]
        destination = Path(request["export"]["filename"]).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        compound = b.Compound(children=list(shown.values()))
        exporter = {"step": b.export_step, "stl": b.export_stl, "brep": b.export_brep}[
            fmt
        ]
        precision = float(request["export"].get("precision", 0.1))
        success = (
            exporter(compound, destination, tolerance=precision)
            if fmt == "stl"
            else exporter(compound, destination)
        )
        if (
            success is False
            or not destination.exists()
            or not destination.stat().st_size
        ):
            raise ValueError("Export did not produce a nonempty file")
        imported = {"step": b.import_step, "stl": b.import_stl, "brep": b.import_brep}[
            fmt
        ](destination)
        if imported.wrapped.IsNull() or (
            fmt != "stl" and (not imported.vertices() or not imported.is_valid)
        ):
            raise ValueError("Exported file could not be reopened as valid geometry")
        before, after = (
            compound.bounding_box().size,
            imported.bounding_box(optimal=fmt != "stl").size,
        )
        tolerance = max(0.01, 2.5 * precision) if fmt == "stl" else 0.01
        error = max(abs(a - c) for a, c in zip(before, after))
        passed = error <= tolerance
        report["rows"].append(
            row(
                f"{fmt.upper()} reopened: overall dimensions",
                "Passed" if passed else "Failed",
                f"max error {error:.6g} mm",
                f"≤ {tolerance:g} mm",
                str(destination),
            )
        )
        if fmt != "stl":
            count_ok = len(compound.solids()) == len(imported.solids())
            report["rows"].append(
                row(
                    "Exported solid count",
                    "Passed" if count_ok else "Failed",
                    str(len(imported.solids())),
                    str(len(compound.solids())),
                )
            )
            volume_error = abs(compound.volume - imported.volume)
            volume_tolerance = max(1e-4, abs(compound.volume) * 1e-5)
            report["rows"].append(
                row(
                    "Exported volume",
                    "Passed" if volume_error <= volume_tolerance else "Failed",
                    f"error {volume_error:.6g} mm³",
                    f"≤ {volume_tolerance:g} mm³",
                )
            )
        else:
            report["rows"].extend(inspect_stl(destination))
        report["ok"] = not any(r["status"] == "Failed" for r in report["rows"])
    simple_locals = {}
    for name, value in namespace.items():
        if name.startswith("_"):
            continue
        if (
            isinstance(value, (str, int, float, bool, type(None)))
            and len(str(value)) < 10000
        ):
            if not isinstance(value, float) or math.isfinite(value):
                simple_locals[name] = value
    extra_shapes = []
    if request.get("auto_show"):
        for name, value in namespace.items():
            if (
                isinstance(value, b.Shape)
                and name not in shown
                and len(extra_shapes) < 100
            ):
                path = folder / f"local-{len(extra_shapes)}.brep"
                b.export_brep(value, path)
                item = dict(name=name, path=str(path))
                if getattr(value, "children", ()):
                    item["tree"] = serialize_tree(value)
                extra_shapes.append(item)
    return dict(
        ok=True,
        objects=manifest,
        report=report,
        locals=simple_locals,
        extra_shapes=extra_shapes,
    )


def worker_main():
    request_path = Path(sys.argv[-1]).resolve()
    folder = request_path.parent
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        workdir = request.get("workdir")
        if workdir:
            import os

            os.chdir(workdir)
            sys.path.insert(0, workdir)
        result = run_job(request, folder)
    except BaseException as error:
        frames = traceback.extract_tb(error.__traceback__)
        line = next(
            (f.lineno for f in reversed(frames) if f.filename == "<cad-worker>"), None
        )
        if isinstance(error, SyntaxError):
            line = error.lineno
        result = dict(
            ok=False,
            error=f"{type(error).__name__}: {error}",
            line=line,
            version=getattr(sys.modules.get("build123d"), "__version__", ""),
            traceback=traceback.format_exc(limit=8)[-6000:],
        )
    (folder / "result.json").write_text(
        json.dumps(result, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    worker_main()
