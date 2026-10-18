"""Run bounded child CAD jobs while Qt continues dispatching UI events.

The guarded caller keeps a synchronous result API; no polling spin or geometry
execution happens on the UI thread. Cancellation terminates the child itself.
"""

import json
import sys
import tempfile
from pathlib import Path

from PyQt5.QtCore import QEventLoop, QProcess, QProcessEnvironment, QTimer


def run_cad_job(
    request, *, owner=None, timeout=180, cancelled=lambda: False, shapes=None
):
    root = owner
    while root is not None and not isinstance(getattr(root, "components", None), dict):
        root = root.parent()
    if root is None and owner is not None:
        root = owner.window()
    if root is not None and getattr(root, "_cad_job_running", False):
        return dict(ok=False, error="Error: another CAD operation is running.")
    if root is not None:
        root._cad_job_running = True
    editor = getattr(root, "components", {}).get("editor") if root is not None else None
    was_readonly = editor.isReadOnly() if editor is not None else False
    editor_snapshot = (
        (editor.get_text_with_eol(), editor.filename) if editor is not None else None
    )
    file_actions = (
        list(getattr(editor, "_actions", {}).get("File", []))
        if editor is not None
        else []
    )
    action_states = [(action, action.isEnabled()) for action in file_actions]
    if editor is not None:
        editor.setReadOnly(True)
    for action, _ in action_states:
        action.setEnabled(False)
    try:
        result = _execute_job(
            request, timeout=timeout, cancelled=cancelled, shapes=shapes
        )
        if result.get("ok") and request.get("code") and editor is not None:
            if editor_snapshot != (editor.get_text_with_eol(), editor.filename):
                return dict(
                    ok=False,
                    error="Error: editor changed during the CAD build; the outdated result was discarded.",
                )
        return result
    finally:
        for action, enabled in action_states:
            action.setEnabled(enabled)
        if editor is not None:
            editor.setReadOnly(was_readonly)
        if root is not None:
            root._cad_job_running = False


def _execute_job(request, *, timeout, cancelled, shapes):
    import build123d as b

    with tempfile.TemporaryDirectory(prefix="cad-agent-job-") as folder:
        folder = Path(folder)
        request = dict(request)
        if shapes is not None:
            request["input_shapes"] = []
            for i, (name, shape) in enumerate(shapes.items()):
                path = folder / f"input-{i}.brep"
                b.export_brep(shape, path)
                request["input_shapes"].append(dict(name=name, path=str(path)))
        request_path = folder / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        loop = QEventLoop()
        process = QProcess()
        process.setProcessChannelMode(QProcess.MergedChannels)
        if getattr(sys, "frozen", False):
            environment = QProcessEnvironment.systemEnvironment()
            environment.insert("PYINSTALLER_SUPPRESS_SPLASH_SCREEN", "1")
            process.setProcessEnvironment(environment)
            arguments = ["--cad-worker", str(request_path)]
        else:
            root = str(Path(__file__).resolve().parents[2])
            bootstrap = (
                "import sys; sys.path.insert(0, "
                + repr(root)
                + "); from cq_editor.ai.cad_runtime import worker_main; worker_main()"
            )
            arguments = ["-c", bootstrap, str(request_path)]
        outcome = {}
        log_tail = bytearray()

        def drain():
            log_tail.extend(bytes(process.readAllStandardOutput()))
            if len(log_tail) > 16000:
                del log_tail[:-16000]

        def finish(*args):
            drain()
            loop.quit()

        def abort(reason):
            outcome["error"] = reason
            process.kill()

        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(
            lambda: abort(
                f"TIMEOUT: CAD job exceeded {timeout}s and was killed. Simplify the design or increase the timeout."
            )
        )
        cancel_timer = QTimer()
        cancel_timer.setInterval(50)
        cancel_timer.timeout.connect(
            lambda: abort("Cancelled: CAD job stopped.") if cancelled() else None
        )
        process.readyReadStandardOutput.connect(drain)
        process.finished.connect(finish)
        process.errorOccurred.connect(
            lambda error: finish() if error == QProcess.FailedToStart else None
        )
        if cancelled():
            return dict(ok=False, error="Cancelled: CAD job stopped.")
        process.start(sys.executable, arguments)
        watchdog.start(max(1, int(timeout * 1000)))
        cancel_timer.start()
        loop.exec_()
        watchdog.stop()
        cancel_timer.stop()
        if process.state() != QProcess.NotRunning:
            process.kill()
            process.waitForFinished(1000)
        if outcome.get("error"):
            return dict(ok=False, **outcome)
        result_path = folder / "result.json"
        if process.exitStatus() != QProcess.NormalExit or not result_path.exists():
            return dict(
                ok=False,
                error=f"CRASH: CAD worker exited ({process.exitCode()}). "
                + log_tail.decode("utf-8", "replace")[-1000:],
            )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))

            def load_tree(node):
                if "children" in node:
                    shape = b.Compound(
                        children=[load_tree(child) for child in node["children"]]
                    )
                    position, rotation = node["location"]
                    shape.location = b.Location(position, rotation)
                else:
                    shape = b.import_brep(node["path"])
                shape.label = node.get("label", "")
                if "color" in node:
                    shape.color = b.Color(*node["color"])
                return shape

            if result.get("ok"):
                for item in result["objects"] + result.get("extra_shapes", []):
                    item["shape"] = (
                        load_tree(item["tree"])
                        if "tree" in item
                        else b.import_brep(item["path"])
                    )
            result["log"] = log_tail.decode("utf-8", "replace")
            return result
        except Exception as error:
            return dict(
                ok=False, error=f"Error: could not load completed CAD job: {error}"
            )
