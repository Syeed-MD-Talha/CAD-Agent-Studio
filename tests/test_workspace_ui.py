"""Desktop layout, theme isolation, and the original CAD action wiring."""

import sys
import os

import pytest
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtGui import QPalette, QFontDatabase
from PyQt5.QtWidgets import QApplication, QMessageBox

from cq_editor.__main__ import MainWindow
from cq_editor.ai.sessions import SessionStore
from cq_editor.preferences import PreferencesWidget
from cq_editor.widgets.occt_widget import OCCTWidget


@pytest.fixture
def workspace(qtbot, tmp_path, monkeypatch):
    if os.name == "nt":
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    monkeypatch.setattr(
        "cq_editor.mixins.QSettings",
        lambda *args: QSettings(str(tmp_path / "app.ini"), QSettings.IniFormat),
    )
    monkeypatch.setattr(
        "cq_editor.widgets.chat.SessionStore",
        lambda: SessionStore(tmp_path / "sessions"),
    )
    monkeypatch.setattr("cq_editor.widgets.chat.load_environment", lambda: None)
    monkeypatch.setattr(MainWindow, "setup_logging", lambda self: None)
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Discard
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes
    )
    # Exercise the real CAD context and widgets; headless Qt cannot create a
    # native Windows OpenGL surface. Hardware rendering is checked separately.
    monkeypatch.setattr(OCCTWidget, "paintEvent", lambda self, event: None)
    original_hook = sys.excepthook
    window = MainWindow()
    window.components["editor"].preferences["Color scheme"] = "Follow app"
    window.preferences["Light/Dark Theme"] = "Light"
    window.resize(1360, 850)
    qtbot.addWidget(window)
    window.show()
    yield window
    window.components["editor"].document().setModified(False)
    window.close()
    console = window.components["console"]
    console.kernel_client.stop_channels()
    console.kernel_manager.shutdown_kernel()
    sys.excepthook = original_hook


def test_responsive_workspace_and_panel_controls(workspace, qtbot):
    chat = workspace.components["chat"]
    assert chat.minimumSizeHint().width() < 500
    assert not chat.sidebar.isVisible()
    assert not workspace.docks["variables_viewer"].isVisible()
    dock = workspace.docks["object_tree"]
    qtbot.mouseClick(dock.titleBarWidget().close_btn, Qt.LeftButton)
    assert dock.isHidden()
    dock.toggleViewAction().trigger()
    assert not dock.isHidden()
    workspace.resize(1100, 760)
    QApplication.processEvents()
    assert workspace.viewport_panel.width() >= 200
    assert chat.provider_combo.width() >= chat.provider_combo.sizeHint().width()


def test_theme_switching_preserves_editor_preference_and_code(workspace):
    editor = workspace.components["editor"]
    code = "from build123d import *\nresult = Box(20, 10, 4)\nshow_object(result)"
    editor.set_text(code)
    editor.document().setModified(False)
    workspace.preferences["Light/Dark Theme"] = "Dark"
    assert editor.palette().color(QPalette.Base).lightness() < 128
    assert not editor.document().isModified()
    assert editor.toPlainText() == code
    editor.preferences["Color scheme"] = "Light"
    assert editor.palette().color(QPalette.Base).lightness() > 128
    assert workspace.toolbar.palette().color(QPalette.Window).lightness() < 128
    workspace.preferences["Light/Dark Theme"] = "Light"
    assert workspace.toolbar.palette().color(QPalette.Window).lightness() > 128
    assert editor.toPlainText() == code


def test_dark_theme_does_not_leave_a_light_editor(workspace):
    editor = workspace.components["editor"]
    editor.preferences["Color scheme"] = "Light"
    workspace.preferences["Light/Dark Theme"] = "Dark"
    assert editor.preferences["Color scheme"] == "Follow app"
    assert editor.palette().color(QPalette.Base).lightness() < 128


def test_render_button_and_model_explorer_keep_original_workflow(workspace):
    editor = workspace.components["editor"]
    editor.set_text(
        "from build123d import *\nresult = Box(20, 10, 4)\nshow_object(result)"
    )
    action = workspace.components["debugger"]._actions["Run"][0]
    assert workspace.render_button.defaultAction() is action
    action.trigger()
    explorer = workspace.components["object_tree"]
    assert explorer.CQ.childCount() == 1
    assert workspace.viewport_panel.object_count.text() == "1 object"
    explorer.CQ.child(0).setSelected(True)
    assert explorer._export_STEP_action.isEnabled()
    assert explorer.properties_editor.isVisible()
    explorer.removeObjects()
    assert workspace.viewport_panel.object_count.text() == "0 objects"


def test_render_errors_are_discoverable(workspace):
    workspace.components["editor"].set_text("this is not valid Python !")
    workspace.components["debugger"]._actions["Run"][0].trigger()
    assert not workspace.docks["traceback_viewer"].isHidden()
    assert workspace.components["traceback_viewer"].current_exception.text()


def test_ai_build_is_reused_for_render_and_reports_wrong_dimension(
    workspace, monkeypatch
):
    from cq_editor.widgets import cad_job

    original = cad_job.run_cad_job
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cad_job, "run_cad_job", counted)
    chat = workspace.components["chat"]
    chat._set_requirements(
        [
            dict(
                kind="extent",
                axis="X",
                expected=31,
                label="Length",
                source="31 mm long",
            )
        ]
    )
    source = "from build123d import *\nresult=Box(30,20,4)\nshow_object(result)"
    assert chat._validate_isolated(source).startswith("OK")
    output = chat.apply_code(source)
    assert output.startswith("Error: design verification failed")
    assert len(calls) == 1
    assert workspace.components["object_tree"].CQ.childCount() == 1
    assert "1 failed" in chat.verification.summary.text()
    summary = chat.verification.summary.text()
    workspace.preferences["Light/Dark Theme"] = "Dark"
    assert chat.verification.summary.text() == summary
    assert chat.verification_report is not None
    workspace.components["editor"].set_text(source + "\n# edited")
    assert chat.verification_report is None
    assert "not checked" in chat.verification.summary.text()


def test_manual_infinite_build_can_be_stopped_without_freezing(workspace, tmp_path):
    import time
    from PyQt5.QtCore import QTimer

    ready = tmp_path / "started.txt"
    source = f"from pathlib import Path\nPath({str(ready)!r}).write_text('running')\nwhile True: pass"
    workspace.components["editor"].set_text(source)
    chat = workspace.components["chat"]
    ticks = []
    timer = QTimer()
    timer.setInterval(30)

    def tick():
        ticks.append(1)
        if ready.exists():
            chat.on_stop()

    timer.timeout.connect(tick)
    timer.start()
    started = time.monotonic()
    workspace.components["debugger"].render()
    timer.stop()
    assert ready.exists()
    assert time.monotonic() - started < 20
    assert len(ticks) > 10
    assert not chat._operation_active
    assert chat.send_btn.isEnabled()
    assert workspace.components["object_tree"].CQ.childCount() == 0


def test_model_tree_export_checks_the_selected_geometry(
    workspace, tmp_path, monkeypatch
):
    from build123d import import_step

    source = "from build123d import *\na=Box(10,10,4)\nb=Pos(30,0,0)*Box(20,10,4)\nshow_object(a,name='a')\nshow_object(b,name='b')"
    workspace.components["editor"].set_text(source)
    workspace.components["debugger"].render()
    tree = workspace.components["object_tree"]
    tree.tree.clearSelection()
    tree.CQ.child(0).setSelected(True)
    destination = tmp_path / "selected.step"
    monkeypatch.setattr(
        "cq_editor.widgets.object_tree.get_save_filename",
        lambda *args: str(destination),
    )
    tree._export_STEP_action.trigger()
    shape = import_step(destination)
    assert len(shape.solids()) == 1
    assert shape.bounding_box().size.X == pytest.approx(10)
    report = workspace.components["chat"].verification_report
    assert any(
        "reopened" in r["label"] and r["status"] == "Passed" for r in report["rows"]
    )


def test_render_discards_build_if_editor_changes_programmatically(workspace):
    from PyQt5.QtCore import QTimer

    editor = workspace.components["editor"]
    editor.set_text("from build123d import *\nshow_object(Box(10,10,4))")
    QTimer.singleShot(100, lambda: editor.set_text("# replacement source"))
    workspace.components["debugger"].render()
    assert editor.get_text_with_eol() == "# replacement source"
    assert workspace.components["object_tree"].CQ.childCount() == 0
    assert "outdated result" in workspace.components["chat"].status.text()
    assert not editor.isReadOnly()


def test_preferences_opens_on_general_and_edits_do_not_reset_code(workspace, qtbot):
    dialog = PreferencesWidget(workspace, workspace.components)
    qtbot.addWidget(dialog)
    assert dialog.preferences_tree.currentItem().text(0) == "General"
    assert (
        dialog.stacked.currentWidget() is dialog.preferences_tree.currentItem().widget
    )
    assert (
        "Follow app"
        in workspace.components["editor"]
        .preferences.child("Color scheme")
        .opts["limits"]
    )
    workspace.preferences["Light/Dark Theme"] = "Dark"
    assert "#0d1422" in dialog.styleSheet()
    workspace.components["editor"].set_text("result = 'keep this code'")
    workspace.reset_workspace_layout()
    assert workspace.components["editor"].toPlainText() == "result = 'keep this code'"


def test_provider_profiles_stay_independent_and_lock_during_requests(workspace):
    from cq_editor.ai.providers import ProviderConfig

    chat = workspace.components["chat"]
    for provider in ("fireworks", "featherless"):
        chat._provider_settings.save(
            ProviderConfig(
                provider,
                provider + "-test-key",
                provider + "-model",
                "https://example.com/v1",
            )
        )
    for provider in ("featherless", "fireworks"):
        chat.provider_combo.setCurrentIndex(chat.provider_combo.findData(provider))
        assert chat._selected_config().api_key == provider + "-test-key"
        assert provider + "-model" in chat.model_lbl.text()
    chat._set_busy(True)
    assert not chat.provider_combo.isEnabled()
    assert not chat.api_settings_btn.isEnabled()
    chat._set_busy(False)
    assert chat.provider_combo.isEnabled()
    assert "test-key" not in chat._transcript_markdown()


def test_provider_dialog_preserves_manual_model_and_masks_key(
    workspace, qtbot, monkeypatch
):
    from cq_editor.ai.providers import ProviderConfig
    from cq_editor.widgets.provider_settings import ProviderSettingsDialog

    monkeypatch.setattr(
        "cq_editor.widgets.provider_settings.list_models",
        lambda *args, **kwargs: [
            "org/Available-A",
            "org/Available-B",
        ],
    )
    dialog = ProviderSettingsDialog(
        ProviderConfig(
            "custom_openai",
            "preview-test-key",
            "org/Manual-model",
            "https://example.com/v1",
        ),
        workspace.components["chat"],
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: "2 models loaded" in dialog.model_status.text())
    assert dialog.model.currentText() == "org/Manual-model"
    assert dialog.api_key.echoMode() == dialog.api_key.Password
    dialog.show_key.setChecked(True)
    assert dialog.api_key.echoMode() == dialog.api_key.Normal
    dialog.model.setCurrentText("org/Available-B")
    dialog.accept()
    assert dialog.result() == dialog.Accepted
    assert dialog.config().model == "org/Available-B"


def test_provider_catalog_failure_keeps_manual_entry(workspace, qtbot, monkeypatch):
    from cq_editor.ai.providers import ProviderConfig
    from cq_editor.widgets.provider_settings import ProviderSettingsDialog

    def unavailable(*args, **kwargs):
        raise RuntimeError("Model listing unavailable. Enter a model ID manually.")

    monkeypatch.setattr("cq_editor.widgets.provider_settings.list_models", unavailable)
    dialog = ProviderSettingsDialog(
        ProviderConfig(
            "custom_openai",
            "preview-test-key",
            "org/Manual-model",
            "https://example.com/v1",
        ),
        workspace.components["chat"],
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: "unavailable" in dialog.model_status.text())
    assert dialog.model.currentText() == "org/Manual-model"
    assert dialog.model.isEnabled()
    assert dialog.refresh_models_btn.isEnabled()
    dialog.reject()
