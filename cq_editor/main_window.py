import sys
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, Qt, QSize, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QDockWidget,
    QAction,
    QApplication,
    QMenu,
    QProxyStyle,
    QStyle,
    QStyleFactory,
    QToolButton,
    QWidget,
    QSizePolicy,
    QTabWidget,
)
from logbook import Logger
import build123d as b3d

from .widgets.editor import Editor
from .widgets.viewer import OCCViewer
from .widgets.console import ConsoleWidget
from .widgets.object_tree import ObjectTree
from .widgets.traceback_viewer import TracebackPane
from .widgets.debugger import Debugger, LocalsView
from .widgets.cq_object_inspector import CQObjectInspector
from .widgets.log import LogViewer
from .widgets.upload_dialog import UploadDialog

try:
    from .widgets.chat import ChatWidget

    _HAS_CHAT = True
except Exception:
    ChatWidget = None  # type: ignore
    _HAS_CHAT = False

from . import __version__
from .utils import (
    dock,
    open_url,
    about_dialog,
    check_gtihub_for_updates,
    confirm,
)
from .mixins import MainMixin
from .icons import icon, refresh_icons
from .theme import apply_application_theme
from .widgets.workspace import DockHeader, ViewportPanel
import qtawesome as qta
from pyqtgraph.parametertree import Parameter
from .preferences import PreferencesWidget


class _PrintRedirectorSingleton(QObject):
    """This class monkey-patches `sys.stdout.write` to emit a signal.
    It is instanciated as `.main_window.PRINT_REDIRECTOR` and should not be instanciated again.
    """

    sigStdoutWrite = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        original_stdout_write = sys.stdout.write

        def new_stdout_write(text: str):
            self.sigStdoutWrite.emit(text)
            return original_stdout_write(text)

        sys.stdout.write = new_stdout_write


PRINT_REDIRECTOR = _PrintRedirectorSingleton()


class DockSeparatorStyle(QProxyStyle):
    """Widens the dock separators so that they are easier to grab (#277).

    A style proxy is used instead of a stylesheet because setting a stylesheet
    on the main window moves its whole widget subtree to QStyleSheetStyle,
    which discards palette based theming (e.g. editor syntax highlighting).
    """

    def pixelMetric(self, metric, option=None, widget=None):

        extent = super().pixelMetric(metric, option, widget)
        if metric == QStyle.PM_DockWidgetSeparatorExtent:
            return max(extent, 6)

        return extent


class MainWindow(QMainWindow, MainMixin):

    name = "CAD Agent Studio"
    org = "CadQuery"

    preferences = Parameter.create(
        name="Preferences",
        children=[
            {
                "name": "Light/Dark Theme",
                "type": "list",
                "value": "Light",
                "values": [
                    "Light",
                    "Dark",
                ],
            },
        ],
    )

    def __init__(self, parent=None, filename=None):

        super(MainWindow, self).__init__(parent)
        MainMixin.__init__(self)
        self.components = {}
        self.docks = {}

        self.setWindowIcon(icon("app"))

        # Windows workaround - makes the correct task bar icon show up.
        if sys.platform == "win32":
            import ctypes

            myappid = "cad-agent-studio"  # arbitrary string
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        self._theme = apply_application_theme(
            self.preferences["Light/Dark Theme"] == "Dark"
        )
        self.setDockNestingEnabled(True)
        self.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)
        self.resize(1440, 920)
        self.viewer = OCCViewer(self)
        self.viewport_panel = ViewportPanel(self.viewer, self)
        self.setCentralWidget(self.viewport_panel)

        # Make sure the dock separators are wide enough to grab on high-DPI displays.
        # setStyle does not take ownership, so keep a reference on self.
        self._separator_style = DockSeparatorStyle(QStyleFactory.create("Fusion"))
        self.setStyle(self._separator_style)

        self.prepare_panes()
        self.registerComponent("viewer", self.viewer)
        self.prepare_toolbar()
        self.prepare_menubar()

        self.prepare_statusbar()
        self.prepare_actions()

        self.components["object_tree"].addLines()

        self.prepare_console()

        self.fill_dummy()

        self.setup_logging()

        # Allows us to react to the top-level settings for this window being changed
        self.preferences.sigTreeStateChanged.connect(self.preferencesChanged)

        self.restorePreferences()
        self.preferencesChanged(None, None)
        self.reset_workspace_layout()
        self.restoreWindow()
        if not self.settings.value("workspace/designVersion"):
            self.reset_workspace_layout()
            self.settings.setValue("workspace/designVersion", 1)

        # Handle the event of the editor being hidden or shown
        self.editor_dock = self.docks["editor"]
        self.editor_dock.visibilityChanged.connect(self.handleEditorVisiblityChange)
        self.handleEditorVisiblityChange(not self.editor_dock.isHidden())

        # Let the user know when the file has been modified
        self.components["editor"].document().modificationChanged.connect(
            self.update_window_title
        )

        if filename:
            self.components["editor"].load_from_file(filename)

        self.restoreComponentState()

        try:
            import pyi_splash

            pyi_splash.close()
        except ImportError:
            pass

    def handleEditorVisiblityChange(self, visible):
        """
        Does the work required to enable/disable menu items when the Editor visibility is changed.
        """
        self.toggle_comment_action.setEnabled(visible)
        self.autocomplete_action.setEnabled(visible)

    def preferencesChanged(self, param, changes):
        """
        Triggered when the preferences for this window are changed.
        """

        dark = self.preferences["Light/Dark Theme"] == "Dark"
        self._theme = apply_application_theme(dark)
        self.setFont(QApplication.font())
        self.components["console"].app_theme_changed("Dark" if dark else "Light")
        for dock_widget in self.docks.values():
            dock_widget.titleBarWidget().apply_theme(self._theme)
        self.viewport_panel.apply_theme(self._theme)
        self.viewer.updatePreferences()
        self.viewport_panel.projection.setText(
            self.viewer.preferences["Projection Type"]
        )
        self.theme_action.blockSignals(True)
        self.theme_action.setChecked(dark)
        self.theme_action.blockSignals(False)
        self.theme_action.setText("Light theme" if dark else "Dark theme")
        refresh_icons(self)
        self.render_button.setIcon(qta.icon("fa5s.play", color="#ffffff"))
        self.debug_button.setIcon(qta.icon("fa5s.bug", color=self._theme.muted))
        self.export_button.setIcon(qta.icon("fa5s.download", color=self._theme.muted))
        self.workspace_button.setIcon(qta.icon("fa5s.columns", color=self._theme.muted))
        self.preferences_button.setIcon(
            qta.icon("fa5s.sliders-h", color=self._theme.muted)
        )
        self.theme_button.setIcon(
            qta.icon("fa5s.sun" if dark else "fa5s.moon", color=self._theme.muted)
        )
        self.toolbar.setPalette(QApplication.palette())
        chat = self.components.get("chat")
        if chat is not None:
            chat._on_app_theme_changed()
        editor = self.components["editor"]
        editor_scheme = editor.preferences["Color scheme"]
        # Avoid a high-contrast mismatch when switching the application theme:
        # a light editor in a dark workspace (or vice versa) is almost always
        # an old saved preference, not an intentional choice.
        opposite = (dark and editor_scheme == "Light") or (
            not dark and editor_scheme == "Dark"
        )
        if opposite:
            editor.preferences["Color scheme"] = "Follow app"
        else:
            editor.set_color_scheme(editor_scheme)

    def closeEvent(self, event):

        chat = self.components.get("chat")
        debugger = self.components.get("debugger")
        workers = getattr(chat, "_active_workers", [])
        if getattr(self, "_cad_job_running", False) or any(
            worker.isRunning() for worker in workers
        ):
            if chat:
                chat._stop_requested = True
                for worker in workers:
                    worker.requestInterruption()
            if debugger:
                debugger.cancel_render()
            self.statusBar().showMessage("Stopping active jobs before closing…")
            event.ignore()
            if not getattr(self, "_close_retry_pending", False):
                self._close_retry_pending = True

                def retry_close():
                    self._close_retry_pending = False
                    self.close()

                QTimer.singleShot(200, retry_close)
            return

        self.saveWindow()
        self.savePreferences()
        self.saveComponentState()

        if self.components["editor"].document().isModified():

            rv = QMessageBox.warning(
                self,
                "Unsaved changes",
                "Save changes before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )

            if rv == QMessageBox.Save:
                self.components["editor"].save()
                if self.components["editor"].document().isModified():
                    # Save As was cancelled - abort the close.
                    event.ignore()
                    return
                event.accept()
                super(MainWindow, self).closeEvent(event)
            elif rv == QMessageBox.Discard:
                event.accept()
                super(MainWindow, self).closeEvent(event)
            else:
                event.ignore()
        else:
            super(MainWindow, self).closeEvent(event)

    def prepare_panes(self):

        self.registerComponent(
            "editor",
            Editor(self),
            lambda c: dock(c, "Editor", self, defaultArea="left"),
        )

        self.registerComponent(
            "object_tree",
            ObjectTree(self),
            lambda c: dock(c, "Objects", self, defaultArea="right"),
        )

        self.registerComponent(
            "traceback_viewer",
            TracebackPane(self),
            lambda c: dock(c, "Current traceback", self, defaultArea="bottom"),
        )

        self.registerComponent("debugger", Debugger(self))

        self.registerComponent(
            "console",
            ConsoleWidget(self),
            lambda c: dock(c, "Console", self, defaultArea="bottom"),
        )

        self.registerComponent(
            "variables_viewer",
            LocalsView(self),
            lambda c: dock(c, "Variables", self, defaultArea="right"),
        )

        self.registerComponent(
            "cq_object_inspector",
            CQObjectInspector(self),
            lambda c: dock(c, "CQ object inspector", self, defaultArea="right"),
        )
        self.registerComponent(
            "log",
            LogViewer(self),
            lambda c: dock(c, "Log viewer", self, defaultArea="bottom"),
        )

        if _HAS_CHAT:
            self.registerComponent(
                "chat",
                ChatWidget(self),
                lambda c: dock(c, "AI Chat", self, defaultArea="right"),
            )

        for d in self.docks.values():
            display_titles = {
                "Editor": "Code editor",
                "Objects": "Model explorer",
                "CQ object inspector": "Shape inspector",
                "Current traceback": "Render errors",
                "Log viewer": "Activity log",
            }
            d.setWindowTitle(display_titles.get(d.objectName(), d.windowTitle()))
            d.setTitleBarWidget(DockHeader(d))
            d.show()

        # Connect the bound method rather than a lambda: PRINT_REDIRECTOR is a
        # module-level singleton, and PyQt drops a connection to a QObject's
        # bound method when that QObject is destroyed. A lambda would outlive
        # the LogViewer and call into a deleted C++ object.
        PRINT_REDIRECTOR.sigStdoutWrite.connect(self.components["log"].append)

    def prepare_menubar(self):

        menu = self.menuBar()

        menu_file = menu.addMenu("&File")
        menu_edit = menu.addMenu("&Edit")
        menu_tools = menu.addMenu("&Tools")
        menu_run = menu.addMenu("&Run")
        menu_view = menu.addMenu("&View")
        menu_help = menu.addMenu("&Help")

        # per component menu elements
        menus = {
            "File": menu_file,
            "Edit": menu_edit,
            "Run": menu_run,
            "Tools": menu_tools,
            "View": menu_view,
            "Help": menu_help,
        }

        for comp in self.components.values():
            self.prepare_menubar_component(menus, comp.menuActions())

        # Examples submenu
        menu_file.addSeparator()
        examples_menu = menu_file.addMenu("Examples")
        self._populate_examples_menu(examples_menu)

        menu_file.addSeparator()
        menu_file.addAction(
            QAction("Quit", self, shortcut="ctrl+Q", triggered=self.close)
        )

        # global menu elements
        menu_view.addSeparator()
        for d in self.findChildren(QDockWidget):
            menu_view.addAction(d.toggleViewAction())

        menu_view.addSeparator()
        for t in self.findChildren(QToolBar):
            menu_view.addAction(t.toggleViewAction())

        self.toggle_comment_action = QAction(
            icon("toggle-comment"),
            "Toggle Comment",
            self,
            shortcut="ctrl+/",
            triggered=self.components["editor"].toggle_comment,
        )
        menu_edit.addAction(self.toggle_comment_action)

        # Add the menu action to toggle auto-completion
        self.autocomplete_action = QAction(
            icon("search"),
            "Auto-Complete",
            self,
            shortcut="alt+/",
            triggered=self.components["editor"]._trigger_autocomplete,
        )
        menu_edit.addAction(self.autocomplete_action)

        # Add the menu action to open the code search controls
        self.search_action = QAction(
            icon("search"),
            "Search",
            self,
            shortcut="ctrl+F",
            triggered=self.components["editor"].search_widget.show_search,
        )
        menu_edit.addAction(self.search_action)

        menu_edit.addAction(
            QAction(
                icon("preferences"),
                "Preferences",
                self,
                triggered=self.edit_preferences,
            )
        )

        menu_help.addAction(
            QAction(icon("help"), "Documentation", self, triggered=self.documentation)
        )

        menu_help.addAction(
            QAction("build123d documentation", self, triggered=self.cq_documentation)
        )

        menu_help.addAction(QAction(icon("about"), "About", self, triggered=self.about))

        menu_help.addAction(
            QAction(
                "Check for build123d updates", self, triggered=self.check_for_cq_updates
            )
        )

        # Add the Upload dialog action
        menu_tools.addAction(
            QAction(
                "Upload Model...",
                self,
                triggered=self._upload_model,
            )
        )

    def prepare_menubar_component(self, menus, comp_menu_dict):

        for name, action in comp_menu_dict.items():
            menus[name].addActions(action)

    def prepare_toolbar(self):
        self.toolbar = QToolBar("Main toolbar", self, objectName="Main toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(18, 18))
        brand_icon = QLabel(self.toolbar)
        brand_icon.setPixmap(
            QPixmap(":/images/icons/cad_studio_mark.png").scaled(
                24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        brand_icon.setObjectName("studioBrandIcon")
        self.toolbar.addWidget(brand_icon)
        brand = QLabel("CAD Agent Studio", self.toolbar)
        brand.setObjectName("studioBrand")
        self.toolbar.addWidget(brand)
        self.toolbar.addSeparator()
        for action in self.components["editor"]._actions["File"][:3]:
            self.toolbar.addAction(action)
        self.toolbar.addSeparator()
        debugger_actions = self.components["debugger"]._actions["Run"]
        self.render_button = QToolButton(self.toolbar)
        self.render_button.setObjectName("renderAction")
        self.render_button.setDefaultAction(debugger_actions[0])
        self.render_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toolbar.addWidget(self.render_button)
        debug_button = QToolButton(self.toolbar)
        self.debug_button = debug_button
        debug_button.setText("Debug")
        debug_button.setIcon(icon("debug"))
        debug_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        debug_button.setPopupMode(QToolButton.InstantPopup)
        debug_menu = QMenu(debug_button)
        debug_menu.addActions(debugger_actions[1:])
        debug_button.setMenu(debug_menu)
        self.toolbar.addWidget(debug_button)
        self.toolbar.addSeparator()
        export_button = QToolButton(self.toolbar)
        self.export_button = export_button
        export_button.setText("Export")
        export_button.setIcon(qta.icon("fa5s.download"))
        export_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        export_button.setPopupMode(QToolButton.InstantPopup)
        export_menu = QMenu(export_button)
        tree = self.components["object_tree"]
        export_menu.addActions([tree._export_STEP_action, tree._export_STL_action])
        export_button.setMenu(export_menu)
        self.toolbar.addWidget(export_button)
        spacer = QWidget(self.toolbar)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)
        workspace = QToolButton(self.toolbar)
        self.workspace_button = workspace
        workspace.setText("Workspace")
        workspace.setIcon(qta.icon("fa5s.columns"))
        workspace.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        workspace.setPopupMode(QToolButton.InstantPopup)
        workspace_menu = QMenu(workspace)
        workspace_menu.addAction("Reset panel layout", self.reset_workspace_layout)
        workspace_menu.addSeparator()
        for dock_widget in self.docks.values():
            workspace_menu.addAction(dock_widget.toggleViewAction())
        workspace.setMenu(workspace_menu)
        self.toolbar.addWidget(workspace)
        self.theme_action = QAction("Dark theme", self, checkable=True)
        self.theme_action.toggled.connect(
            lambda checked: self.preferences.__setitem__(
                "Light/Dark Theme", "Dark" if checked else "Light"
            )
        )
        self.theme_button = QToolButton(self.toolbar)
        self.theme_button.setDefaultAction(self.theme_action)
        self.theme_button.setToolTip("Switch light / dark theme")
        self.toolbar.addWidget(self.theme_button)
        preferences_button = QToolButton(self.toolbar)
        self.preferences_button = preferences_button
        preferences_button.setIcon(icon("preferences"))
        preferences_button.setToolTip("Preferences")
        preferences_button.clicked.connect(self.edit_preferences)
        self.toolbar.addWidget(preferences_button)
        self.addToolBar(self.toolbar)

    def reset_workspace_layout(self):
        """A focused default; existing saved arrangements are restored after it."""
        for panel in self.docks.values():
            panel.setFloating(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.docks["editor"])
        self.splitDockWidget(
            self.docks["editor"], self.docks["object_tree"], Qt.Vertical
        )
        for name in ("variables_viewer", "cq_object_inspector"):
            self.tabifyDockWidget(self.docks["object_tree"], self.docks[name])
            self.docks[name].hide()
        for name in ("console", "log", "traceback_viewer"):
            self.addDockWidget(Qt.BottomDockWidgetArea, self.docks[name])
        self.tabifyDockWidget(self.docks["console"], self.docks["log"])
        self.tabifyDockWidget(self.docks["console"], self.docks["traceback_viewer"])
        for name in ("console", "log", "traceback_viewer"):
            self.docks[name].hide()
        if "chat" in self.docks:
            self.addDockWidget(Qt.RightDockWidgetArea, self.docks["chat"])
            self.docks["chat"].show()
            self.resizeDocks([self.docks["chat"]], [480], Qt.Horizontal)
            self.docks["chat"].raise_()
        self.docks["editor"].show()
        self.docks["object_tree"].show()
        self.docks["object_tree"].raise_()
        self.resizeDocks([self.docks["editor"]], [380], Qt.Horizontal)
        self.resizeDocks(
            [self.docks["editor"], self.docks["object_tree"]], [520, 200], Qt.Vertical
        )

    def prepare_statusbar(self):

        self.status_label = QLabel("Ready", parent=self)
        self.statusBar().addWidget(self.status_label, 1)
        self.cursor_status = QLabel("Ln 1, Col 1", self)
        self.statusBar().addPermanentWidget(self.cursor_status)
        self.statusBar().addPermanentWidget(QLabel("build123d  ·  Python", self))

    def prepare_actions(self):

        if "chat" in self.components:
            self.components["debugger"].sigVerification.connect(
                self.components["chat"]._show_verification
            )
            self.components["editor"].textChanged.connect(
                self.components["chat"]._on_editor_changed
            )

        self.components["debugger"].sigRendered.connect(
            self.components["object_tree"].addObjects
        )
        self.components["debugger"].sigTraceback.connect(
            self.components["traceback_viewer"].addTraceback
        )
        self.components["debugger"].sigLocals.connect(
            self.components["variables_viewer"].update_frame
        )
        self.components["debugger"].sigLocals.connect(
            self.components["console"].push_vars
        )

        self.components["object_tree"].sigObjectsAdded[list].connect(
            self._update_object_count
        )
        self.components["object_tree"].sigObjectsAdded[list, bool].connect(
            self._update_object_count
        )
        self.components["object_tree"].sigObjectsRemoved.connect(
            self._update_object_count
        )
        self.components["debugger"].sigTraceback.connect(self._show_render_error)
        self.components["object_tree"].sigObjectsAdded[list].connect(
            self.components["viewer"].display_many
        )
        self.components["object_tree"].sigObjectsAdded[list, bool].connect(
            self.components["viewer"].display_many
        )
        self.components["object_tree"].sigItemChanged.connect(
            self.components["viewer"].update_item
        )
        self.components["object_tree"].sigObjectsRemoved.connect(
            self.components["viewer"].remove_items
        )
        self.components["object_tree"].sigCQObjectSelected.connect(
            self.components["cq_object_inspector"].setObject
        )
        self.components["object_tree"].sigObjectPropertiesChanged.connect(
            self.components["viewer"].redraw
        )
        self.components["object_tree"].sigAISObjectsSelected.connect(
            self.components["viewer"].set_selected
        )
        self.components["object_tree"].sigHelpersResized.connect(
            self.components["viewer"].redisplay
        )

        self.components["viewer"].sigObjectSelected.connect(
            self.components["object_tree"].handleGraphicalSelection
        )

        self.components["traceback_viewer"].sigHighlightLine.connect(
            self.components["editor"].go_to_line
        )

        self.components["cq_object_inspector"].sigDisplayObjects.connect(
            self.components["viewer"].display_many
        )
        self.components["cq_object_inspector"].sigRemoveObjects.connect(
            self.components["viewer"].remove_items
        )
        self.components["cq_object_inspector"].sigShowPlane.connect(
            self.components["viewer"].toggle_grid
        )
        self.components["cq_object_inspector"].sigShowPlane[bool, float].connect(
            self.components["viewer"].toggle_grid
        )
        self.components["cq_object_inspector"].sigChangePlane.connect(
            self.components["viewer"].set_grid_orientation
        )

        self.components["debugger"].sigLocalsChanged.connect(
            self.components["variables_viewer"].update_frame
        )
        self.components["debugger"].sigLineChanged.connect(
            self.components["editor"].set_debug_line
        )
        self.components["debugger"].sigDebugging.connect(
            self.components["object_tree"].stashObjects
        )
        self.components["debugger"].sigDebugging.connect(
            lambda active: (
                self.components["editor"].clear_debug_line() if not active else None
            )
        )
        self.components["debugger"].sigCQChanged.connect(
            self.components["object_tree"].addObjects
        )
        self.components["debugger"].sigTraceback.connect(
            self.components["traceback_viewer"].addTraceback
        )

        # trigger re-render when file is modified externally or saved
        self.components["editor"].triggerRerender.connect(
            self.components["debugger"].render
        )
        self.components["editor"].sigFilenameChanged.connect(
            self.handle_filename_change
        )
        # Allows updating of the status bar from the Editor
        self.components["editor"].statusChanged.connect(self.update_statusbar)
        self.components["editor"].cursorPositionChanged.connect(
            self._update_cursor_status
        )

    def _update_cursor_status(self):
        cursor = self.components["editor"].textCursor()
        self.cursor_status.setText(
            f"Ln {cursor.blockNumber() + 1}, Col {cursor.positionInBlock() + 1}"
        )

    def _update_object_count(self, *_):
        count = self.components["object_tree"].CQ.childCount()
        self.viewport_panel.object_count.setText(
            f"{count} object{'s' if count != 1 else ''}"
        )

    def _show_render_error(self, exc_info, _code):
        if exc_info:
            self.docks["traceback_viewer"].show()
            self.docks["traceback_viewer"].raise_()

    def prepare_console(self):

        console = self.components["console"]
        obj_tree = self.components["object_tree"]

        # application related items
        console.push_vars({"self": self})

        # CQ related items
        console.push_vars(
            {
                "show": obj_tree.addObject,
                "show_object": obj_tree.addObject,
                "rand_color": self.components["debugger"]._rand_color,
                "b3d": b3d,
                "build123d": b3d,
                "log": Logger(self.name).info,
            }
        )

    def _examples_dir(self):
        # In a PyInstaller bundle examples are extracted alongside the package.
        # In development they live next to the cq_editor package directory.
        if getattr(sys, "frozen", False):
            return Path(sys._MEIPASS) / "examples"
        return Path(__file__).parent.parent / "examples"

    def _populate_examples_menu(self, menu):
        examples_dir = self._examples_dir()
        if not examples_dir.is_dir():
            menu.setEnabled(False)
            return

        for path in sorted(examples_dir.glob("*.py")):
            # Strip the leading "NN_" numbering prefix for the menu label.
            label = path.stem
            if len(label) > 3 and label[2] == "_" and label[:2].isdigit():
                label = label[3:]
            label = label.replace("_", " ").title()

            menu.addAction(
                QAction(
                    label,
                    self,
                    triggered=lambda checked, p=path: self.components[
                        "editor"
                    ].load_example(str(p)),
                )
            )

    def fill_dummy(self):

        self.components["editor"].set_text(
            "from build123d import *\nresult = Box(3, 3, 0.5)\nshow_object(result)"
        )

    def setup_logging(self):

        from logbook.compat import redirect_logging
        from logbook import INFO, Logger

        redirect_logging()
        self.components["log"].handler.level = INFO
        self.components["log"].handler.push_application()

        self._logger = Logger(self.name)

        def handle_exception(exc_type, exc_value, exc_traceback):

            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            self._logger.error(
                "Uncaught exception occurred",
                exc_info=(exc_type, exc_value, exc_traceback),
            )

        sys.excepthook = handle_exception

    def edit_preferences(self):

        prefs = PreferencesWidget(self, self.components)
        prefs.exec_()

    def about(self):

        about_dialog(
            self,
            "About CAD Agent Studio",
            f"AI-assisted build123d CAD workspace.\nVersion: {__version__}",
        )

    def check_for_cq_updates(self):

        check_gtihub_for_updates(self, b3d, github_org="gumyr", github_proj="build123d")

    def documentation(self):

        open_url("https://github.com/CadQuery")

    def cq_documentation(self):

        open_url("https://build123d.readthedocs.io/en/latest/")

    def handle_filename_change(self, fname):

        new_title = fname if fname else "*"
        self.setWindowTitle(f"{self.name}: {new_title}")

    def update_window_title(self, modified):
        """
        Allows updating the window title to show that the document has been modified.
        """
        title = self.windowTitle().rstrip("*")
        if modified:
            title += "*"
        self.setWindowTitle(title)

    def update_statusbar(self, status_text):
        """
        Allow updating the status bar with information.
        """

        # Update the statusbar text
        self.status_label.setText(status_text)

    def _upload_model(self):
        """
        Allows the userr to easily upload models to an online service for manufacturing,
        analysis, simulation, display, etc.
        """

        obj_tree = self.components["object_tree"]
        selected = [
            item
            for item in obj_tree.tree.selectedItems()
            if item.parent() is obj_tree.CQ
        ]
        dlg = UploadDialog(
            self,
            self.components["object_tree"],
            self.components["editor"],
            selected_shapes=[item.shape for item in selected],
        )
        dlg.exec_()


if __name__ == "__main__":

    pass
