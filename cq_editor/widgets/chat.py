"""AI CAD chatbot dock widget for CQ-editor.

Agentic loop (from learn-claude-code s01/s02) split across Qt signals
so the UI stays responsive:
  GUI thread orchestrates -> LLMWorker/QThread does network I/O ->
  GUI executes CAD tools -> repeat until model stops or max_iters.

Tools (executed in GUI thread, Qt-safe):
  get_editor_code / validate_cad_code / apply_and_render / get_examples
  / export_model
"""

import base64
import html
import traceback as tb
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import (
    Qt,
    QThread,
    QTimer,
    QStringListModel,
    QSettings,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QIcon, QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QWidget,
    QTextBrowser,
    QPlainTextEdit,
    QPushButton,
    QCheckBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QMenu,
    QCompleter,
    QHBoxLayout,
    QVBoxLayout,
    QAction,
    QFrame,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
    QApplication,
)
from pyqtgraph.parametertree import Parameter

try:
    import qtawesome as qta

    _HAS_QTA = True
except Exception:
    _HAS_QTA = False

from ..mixins import ComponentMixin
from ..icons import icon
from ..utils import get_save_filename
from ..ai.cad_agent import (
    SYSTEM_PROMPT,  # noqa: F401 (documents prompt source)
    TOOLS,  # noqa: F401
    EFFORTS,
    DEFAULT_EFFORT,
    normalize_effort,
    max_tokens_for_effort,
    extract_python_code,
    build_client,
    headless_validate,
    isolated_validate,
    ISOLATED_PROBE_TIMEOUT,
    build_system_prompt,
)
from ..ai.skill_store import SkillStore
from ..ai.subagent import run_subagent, run_code_subagent
from ..ai.sessions import SessionStore, autotitle, DEFAULT_TITLE, is_auto_title
from ..ai.providers import PROVIDERS, ProviderSettings, load_environment
from .provider_settings import ProviderSettingsDialog, ProviderComboBox
from .controls import CheckBox, ElidedLabel
from .verification import VerificationPanel
from ..ai.verification import normalize_requirements, report_text
from ..theme import theme_for

# Legacy starter snippet (kept for backward compatibility / tests).
# NOTE: new sessions no longer insert this — every fresh chat starts with
# an EMPTY editor and an EMPTY 3D board (nothing auto-drawn, nothing
# auto-rendered). The user (or the agent via apply_and_render) explicitly
# creates the first design.
STARTER_CODE = (
    "from build123d import *\n" "result = Box(3, 3, 0.5)\n" "show_object(result)"
)

# What a brand-new / design-less session shows: truly empty.
EMPTY_DESIGN = ""


# Slash commands (Claude Code / Codex style: type "/" in the composer).
# All are argument-free; Enter (or picking the completion) runs them locally.
SLASH_COMMANDS = {
    "/new": "Start a new chat session",
    "/status": "Show model, mode, session and context stats",
    "/usage": "Show token usage for this chat",
    "/plan": "Toggle plan mode (validate only, never auto-apply)",
    "/undo": "Restore editor code from before the last apply",
    "/export": "Save this chat as a Markdown transcript",
    "/examples": "List bundled build123d examples",
    "/skills": "List CAD skills, or read one: /skills [name] [draft]",
    "/compact": "Show what auto-compaction sends vs stores",
    "/model": "Show the selected provider and model",
    "/effort": "Show or set reasoning effort: /effort [low|medium|high|max]",
    "/help": "List all slash commands",
}


class LLMWorker(QThread):
    """One LLM turn in a background thread (network only, no Qt access)."""

    finished_ok = pyqtSignal(object)  # Anthropic response
    finished_err = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        client,
        model,
        messages,
        parent=None,
        thinking=None,
        max_tokens=8000,
        tools=None,
        skills=None,
        learning=True,
    ):
        super().__init__(parent)
        self.client = client
        self.model = model
        # Shallow copy; blocks are read-only during the call.
        self.messages = list(messages)
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.tools = TOOLS if tools is None else tools
        self.skills = skills
        self.learning = learning

    def run(self):
        try:
            from ..ai.cad_agent import SYSTEM_PROMPT, TOOLS
            from ..ai.recovery import complete

            resp = complete(
                self.client,
                model=self.model,
                system=build_system_prompt(self.skills, learning=self.learning),
                messages=self.messages,
                tools=self.tools,
                max_tokens=self.max_tokens,
                thinking=self.thinking,
                on_status=self.status_changed.emit,
                cancelled=self.isInterruptionRequested,
            )
            self.finished_ok.emit(resp)
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")


class StreamWorker(LLMWorker):
    """Streaming LLM turn: emits text AND thinking deltas live, then final.

    Uses client.messages.stream() so the chat shows tokens as they arrive.
    Iterates parsed stream events (not just text_stream) so extended-thinking
    models (e.g. glm with a thinking budget) show their reasoning live too.
    Falls back to non-streaming create() when the gateway does not support
    streaming (some OpenAI-compatible endpoints).
    """

    text_delta = pyqtSignal(str)
    thinking_delta = pyqtSignal(str)
    stream_reset = pyqtSignal()

    def run(self):
        try:
            from cq_editor.ai.cad_agent import SYSTEM_PROMPT, TOOLS
            from ..ai.recovery import complete

            final = complete(
                self.client,
                model=self.model,
                system=build_system_prompt(self.skills, learning=self.learning),
                messages=self.messages,
                tools=self.tools,
                max_tokens=self.max_tokens,
                thinking=self.thinking,
                stream=True,
                on_text=self.text_delta.emit,
                on_thinking=self.thinking_delta.emit,
                on_reset=self.stream_reset.emit,
                on_status=self.status_changed.emit,
                cancelled=self.isInterruptionRequested,
            )
            self.finished_ok.emit(final)
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")


class ChatWidget(QWidget, ComponentMixin):
    name = "AI Chat"

    preferences = Parameter.create(
        name="Preferences",
        children=[
            {
                "name": "Max tool iterations",
                "type": "int",
                "value": 6,
                "limits": (1, 20),
            },
            {"name": "Auto-apply code", "type": "bool", "value": True},
            {"name": "Learn reusable CAD skills", "type": "bool", "value": True},
            {
                "name": "Pre-render timeout (s)",
                "type": "int",
                "value": 180,
                "limits": (10, 1800),
            },
        ],
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.preferences = Parameter.create(**type(self).preferences.saveState())
        ComponentMixin.__init__(self)

        self.messages = []  # Anthropic conversation history
        self.last_code = ""
        self.worker = None
        self._active_workers = []
        self._operation_active = False
        self._validated_job = None
        self._requirements_recorded_this_turn = False
        self.design_requirements = []
        self.verification_report = None
        self._verification_code = None
        self._visual_capture_revision = None
        self.iters_left = 0
        self._finalizing = False
        self._stop_requested = False
        self._request_rendered = False
        self._pending_traceback = None
        self._stream_buf = ""
        self._think_buf = ""
        self._stream_active = False
        self._stream_start_pos = None
        self._doc_parts = []  # full history HTML; view rebuilt from it
        self._code_snippets = []  # code blocks for [Copy] anchors, in doc order
        self.pending_images = []  # [{name, bytes(JPEG), w, h}] attached, not yet sent
        self._busy_timer = QTimer(self)
        self._busy_timer.setInterval(450)
        self._busy_timer.timeout.connect(self._tick_busy)
        self._busy_dots = 0
        # Coalesce per-token stream refreshes: rebuilding the whole chat
        # document on every delta is O(n^2) and stalls the UI on long
        # reasoning (High/Max effort). Deltas accumulate, view refreshes
        # at most every STREAM_FLUSH_MS, plus a final flush on finish.
        self._stream_timer = QTimer(self)
        self._stream_timer.setInterval(120)
        self._stream_timer.timeout.connect(self._flush_stream_slot)
        self._stream_dirty = False
        # Sessions (s08-transcript + s09-store style, see ai/sessions.py).
        self._store = SessionStore()
        self._skills = SkillStore()
        self.session_id = None
        self.session_title = DEFAULT_TITLE
        self._loading_combo = False
        # Agent-UX state (Claude/Codex patterns).
        self.plan_mode = False
        self.code_history = []  # editor checkpoints for /undo (max 20)
        self.sent_history = []  # composer history for Up/Down
        self._hist_idx = None
        self.sess_usage = {"in": 0, "out": 0}
        self.last_usage = {"in": 0, "out": 0}
        self._turn_steps = 0
        # Design owned by the active session (editor code, re-rendered on switch).
        self.session_design = ""
        self._auto_apply_verdict = ""
        load_environment()
        settings = getattr(parent, "settings", None)
        if settings is None:
            settings = QSettings("CadQuery", "CQ-Editor")
        self._provider_settings = ProviderSettings(settings)

        # ---------- left sidebar: sessions ----------
        self.sidebar = QFrame(self)
        self.sidebar.setObjectName("chatSidebar")
        sidebar_lay = QVBoxLayout(self.sidebar)
        sidebar_lay.setContentsMargins(8, 10, 8, 10)
        sidebar_lay.setSpacing(8)
        sidebar_header = QHBoxLayout()
        sidebar_header.setSpacing(6)
        self.sidebar_title = QLabel("Chats", self.sidebar)
        self.sidebar_title.setObjectName("sidebarTitle")
        sidebar_header.addWidget(self.sidebar_title, 1)
        self.sidebar_new_btn = QPushButton(self.sidebar)
        self.sidebar_new_btn.setObjectName("iconBtn")
        self.sidebar_new_btn.setToolTip("Start new chat")
        self.sidebar_new_btn.setFixedSize(28, 28)
        if _HAS_QTA:
            try:
                self.sidebar_new_btn.setIcon(qta.icon("fa5s.plus", color="#2563eb"))
            except Exception:
                self.sidebar_new_btn.setText("+")
        else:
            self.sidebar_new_btn.setText("+")
        sidebar_header.addWidget(self.sidebar_new_btn)
        sidebar_lay.addLayout(sidebar_header)
        self.session_list = QListWidget(self.sidebar)
        self.session_list.setObjectName("sessionList")
        self.session_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.session_list.setSpacing(4)
        sidebar_lay.addWidget(self.session_list, 1)
        self.sidebar.setFixedWidth(170)

        # ---------- main chat area ----------
        chat_area = QVBoxLayout()
        chat_area.setContentsMargins(0, 0, 0, 0)
        chat_area.setSpacing(8)

        # header
        header = QFrame(self)
        header.setObjectName("chatHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 12, 12, 12)
        header_layout.setSpacing(12)
        hlay = QHBoxLayout()
        hlay.setContentsMargins(0, 0, 0, 0)
        hlay.setSpacing(10)
        self.title_icon = QLabel(header)
        self.title_icon.setObjectName("brandIcon")
        self.title_icon.setPixmap(
            QPixmap(":/images/icons/cad_studio_mark.png").scaled(
                26, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel("CAD Assistant", header)
        self.title_label.setObjectName("chatTitle")
        self.subtitle_label = QLabel("Describe. Refine. Create.", header)
        self.subtitle_label.setObjectName("chatSubtitle")
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.subtitle_label)
        self.mode_pill = QLabel("PLAN", header)
        self.mode_pill.setObjectName("modePill")
        self.mode_pill.setToolTip(
            "Plan mode: code is validated, never auto-applied (/plan to toggle)"
        )
        self.mode_pill.setVisible(False)
        hlay.addWidget(self.title_icon)
        hlay.addLayout(title_box, 1)
        hlay.addWidget(self.mode_pill)
        self.chats_btn = QPushButton("Chats", header)
        self.chats_btn.setObjectName("iconBtn")
        self.chats_btn.setCheckable(True)
        self.chats_btn.setToolTip("Show saved conversations")
        self.chats_btn.setCursor(Qt.PointingHandCursor)
        self.chats_btn.toggled.connect(self.sidebar.setVisible)
        hlay.addWidget(self.chats_btn)
        self.provider_control = QFrame(header)
        self.provider_control.setObjectName("providerControl")
        provider_row = QHBoxLayout(self.provider_control)
        provider_row.setContentsMargins(2, 2, 2, 2)
        provider_row.setSpacing(2)
        self.provider_combo = ProviderComboBox(self.provider_control)
        self.provider_combo.setObjectName("providerSelect")
        self.provider_combo.setAccessibleName("AI provider")
        self.provider_combo.setMinimumWidth(145)
        self.provider_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        for provider_id, spec in PROVIDERS.items():
            self.provider_combo.addItem(spec.label, provider_id)
        self.provider_combo.setCurrentIndex(
            self.provider_combo.findData(self._provider_settings.selected())
        )
        provider_row.addWidget(self.provider_combo, 1)
        divider = QFrame(self.provider_control)
        divider.setObjectName("providerDivider")
        divider.setFixedSize(1, 20)
        provider_row.addWidget(divider)
        self.api_settings_btn = QPushButton("Settings", self.provider_control)
        self.api_settings_btn.setObjectName("providerSettingsBtn")
        self.api_settings_btn.setToolTip("Manage this provider's API key and model")
        self.api_settings_btn.setCursor(Qt.PointingHandCursor)
        self.api_settings_btn.setFixedHeight(36)
        provider_row.addWidget(self.api_settings_btn)
        self.clear_btn = QPushButton(header)
        self.clear_btn.setObjectName("iconBtn")
        self.clear_btn.setToolTip("Start new chat")
        self.clear_btn.setFixedSize(30, 30)
        if _HAS_QTA:
            try:
                self.clear_btn.setIcon(qta.icon("fa5s.plus", color="#2563eb"))
            except Exception:
                self.clear_btn.setText("✕")
        else:
            self.clear_btn.setText("✕")
        hlay.addWidget(self.clear_btn)
        header_layout.addLayout(hlay)
        header_layout.addWidget(self.provider_control)
        chat_area.addWidget(header)

        # history
        self.history = QTextBrowser(self)
        self.history.setObjectName("chatHistory")
        self.history.setOpenExternalLinks(False)
        self.history.anchorClicked.connect(self._on_history_anchor)
        chat_area.addWidget(self.history, 1)

        # attachment previews (hidden until images attached)
        self.preview_bar = QFrame(self)
        self.preview_bar.setObjectName("previewBar")
        preview_lay = QHBoxLayout(self.preview_bar)
        preview_lay.setContentsMargins(4, 4, 4, 4)
        preview_lay.setSpacing(6)
        self.preview_hint = QLabel(
            "Reference images — click one to remove:", self.preview_bar
        )
        self.preview_hint.setObjectName("muted")
        preview_lay.addWidget(self.preview_hint)
        self.preview_row = QHBoxLayout()
        self.preview_row.setSpacing(6)
        preview_lay.addLayout(self.preview_row, 1)
        self.preview_bar.setVisible(False)
        chat_area.addWidget(self.preview_bar)

        # composer
        self.attach_btn = QPushButton(self)
        self.attach_btn.setObjectName("attachBtn")
        self.attach_btn.setToolTip(
            "Attach or paste a reference image (Ctrl+V — max 3 per message)"
        )
        self.attach_btn.setFixedSize(38, 38)
        if _HAS_QTA:
            try:
                self.attach_btn.setIcon(qta.icon("fa5s.paperclip"))
            except Exception:
                self.attach_btn.setText("📎")
        else:
            self.attach_btn.setText("📎")
        self.input = QPlainTextEdit(self)
        self.input.setObjectName("chatInput")
        self.input.setPlaceholderText(
            "Describe a part…\nPaste an image with Ctrl+V · Enter sends · Shift+Enter adds a line"
        )
        self.input.setMaximumBlockCount(12)
        self.input.setFixedHeight(64)
        self.input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.input.installEventFilter(self)
        self.send_btn = QPushButton("Send", self)
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.setFixedHeight(38)
        if _HAS_QTA:
            try:
                self.send_btn.setIcon(qta.icon("fa5s.paper-plane"))
            except Exception:
                pass
        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setFixedHeight(38)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop the current request / agent loop.")
        composer = QHBoxLayout()
        composer.setSpacing(8)
        composer.addWidget(self.attach_btn)
        composer.addWidget(self.input, 1)
        composer.addWidget(self.send_btn)
        composer.addWidget(self.stop_btn)
        chat_area.addLayout(composer)

        # primary actions
        self.apply_btn = QPushButton("Insert + render", self)
        self.apply_btn.setObjectName("applyBtn")
        if _HAS_QTA:
            try:
                self.apply_btn.setIcon(qta.icon("fa5s.cube"))
            except Exception:
                pass
        self.apply_btn.setEnabled(False)
        self.apply_btn.setToolTip(
            "Write the last generated code into the editor and render it."
        )
        self.undo_btn = QPushButton("Undo", self)
        self.undo_btn.setObjectName("expBtn")
        if _HAS_QTA:
            try:
                self.undo_btn.setIcon(qta.icon("fa5s.undo"))
            except Exception:
                pass
        self.undo_btn.setEnabled(False)
        self.undo_btn.setToolTip(
            "Restore editor code from before the last apply (/undo)."
        )
        self.export_stl_btn = QPushButton("STL", self)
        self.export_stl_btn.setObjectName("expBtn")
        self.export_step_btn = QPushButton("STEP", self)
        self.export_step_btn.setObjectName("expBtn")
        self.export_stl_btn.setToolTip("Export current models to STL.")
        self.export_step_btn.setToolTip("Export current models to STEP.")
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self.apply_btn, 1)
        action_row.addWidget(self.undo_btn)
        action_row.addWidget(self.export_stl_btn)
        action_row.addWidget(self.export_step_btn)
        chat_area.addLayout(action_row)

        # secondary toggles
        self.autofix = CheckBox("Self-fix", self)
        self.autofix.setChecked(True)
        self.autofix.setToolTip(
            "Automatically repair failed tools or code. Successful tools continue with this off."
        )
        self.plan_check = CheckBox("Plan", self)
        self.plan_check.setChecked(False)
        self.plan_check.setToolTip(
            "Plan mode: validate code, never auto-apply (/plan)."
        )
        self.stream_check = CheckBox("Stream", self)
        self.stream_check.setChecked(True)
        self.stream_check.setToolTip(
            "Show tokens as they arrive (falls back if unsupported)."
        )
        self.vision_check = CheckBox("Vision", self)
        self.vision_check.setChecked(True)
        self.vision_check.setToolTip(
            "Visual feedback: after each successful render, send a 3D screenshot "
            "to the model so it can verify and refine the design."
        )
        self.effort = DEFAULT_EFFORT
        self.effort_combo = ProviderComboBox(self)
        self.effort_combo.setMinimumContentsLength(6)
        self.effort_combo.setMinimumHeight(28)
        self.effort_combo.addItems(["Low", "Medium", "High", "Max"])
        self.effort_combo.setCurrentText("Medium")
        self.effort_combo.setToolTip(self._effort_tooltip())
        self.model_lbl = ElidedLabel("Model: choose in API settings", self)
        self.model_lbl.setObjectName("muted")
        self.model_lbl.setToolTip("Change the model and API key in API settings.")
        toggles_row = QHBoxLayout()
        toggles_row.setSpacing(10)
        toggles_row.addWidget(self.effort_combo)
        toggles_row.addWidget(self.autofix)
        toggles_row.addWidget(self.plan_check)
        toggles_row.addWidget(self.stream_check)
        toggles_row.addWidget(self.vision_check)
        toggles_row.addStretch()
        chat_area.addWidget(self.model_lbl)
        chat_area.addLayout(toggles_row)

        self.verification = VerificationPanel(self)
        self.verification.verify_requested.connect(self.on_verify_design)
        self.verification.requirements_changed.connect(self._set_requirements)
        chat_area.addWidget(self.verification)

        # status
        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        self.status_dot = QLabel("●", self)
        self.status_dot.setObjectName("statusDot")
        self.status = QLabel("Choose a provider and open API settings to start.", self)
        self.status.setObjectName("statusLbl")
        self.status.setWordWrap(True)
        self.meter_lbl = QLabel("", self)
        self.meter_lbl.setObjectName("muted")
        self.meter_lbl.setMinimumWidth(0)
        self.meter_lbl.setWordWrap(True)
        self.meter_lbl.setToolTip(
            "Context estimate · last-turn tokens · session totals (/usage)"
        )
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.meter_lbl)
        chat_area.addLayout(status_row)

        # root layout
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        root.addWidget(self.sidebar)
        root.addLayout(chat_area, 1)
        self.setLayout(root)
        self.sidebar.hide()

        self._apply_styles()
        self._welcome()

        self.send_btn.clicked.connect(self.on_send)
        self.stop_btn.clicked.connect(self.on_stop)
        self.clear_btn.clicked.connect(self.on_new_session)
        self.attach_btn.clicked.connect(self.on_attach)
        self.sidebar_new_btn.clicked.connect(self.on_new_session)
        self.session_list.itemClicked.connect(self.on_session_list_clicked)
        self.session_list.itemChanged.connect(self.on_session_item_changed)
        self.session_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.session_list.customContextMenuRequested.connect(self._session_context_menu)
        self.undo_btn.clicked.connect(self.on_undo)
        self.plan_check.toggled.connect(self.on_plan_toggled)
        self.effort_combo.currentTextChanged.connect(self.on_effort_changed)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.api_settings_btn.clicked.connect(self._edit_provider_settings)
        self._refresh_provider_controls()
        self._refresh_model_lbl()
        # Slash-command completion (Claude/Codex "/" popup pattern).
        self.slash_completer = QCompleter(self)
        self.slash_completer.setModel(
            QStringListModel(
                [f"{cmd}  —  {desc}" for cmd, desc in SLASH_COMMANDS.items()]
            )
        )
        self.slash_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.slash_completer.setCompletionMode(QCompleter.PopupCompletion)
        self.slash_completer.setWidget(self.input)
        self.slash_completer.activated[str].connect(self._accept_slash)
        self.input.textChanged.connect(self._update_slash_popup)
        self.apply_btn.clicked.connect(self.on_apply_last)
        self.export_stl_btn.clicked.connect(lambda: self.on_export_button("stl"))
        self.export_step_btn.clicked.connect(lambda: self.on_export_button("step"))

        self._actions = {
            "View": [
                QAction(
                    icon("search"),
                    "AI Chat",
                    self,
                    triggered=self._show_self,
                )
            ]
        }

    @pyqtSlot()
    def _show_self(self):
        """Reveal our own dock (resolves MainWindow lazily - parent is the dock)."""
        try:
            main = self._main()
            if main is not None:
                dock = main.docks.get("chat", None)
                if dock is not None:
                    dock.show()
                    dock.raise_()
        except Exception:
            pass

    # ------------- styling -------------
    def _is_dark(self):
        try:
            from PyQt5.QtWidgets import QApplication

            bg = QApplication.palette().color(self.backgroundRole())
            return (bg.red() * 0.299 + bg.green() * 0.587 + bg.blue() * 0.114) < 128
        except Exception:
            return False

    def _app_is_dark(self):
        """Source of truth: CQ-editor's own Light/Dark preference.

        The old palette-luminance guess misfires (it painted the panel dark
        navy while the app was in Light mode). Fall back to it only when the
        main window is unreachable (e.g. unit tests). Unit tests can force a
        theme via self._theme_override.
        """
        if getattr(self, "_theme_override", None) in ("dark", "light"):
            return self._theme_override == "dark"
        try:
            main = self._main()
            if main is not None:
                theme = main.preferences["Light/Dark Theme"]
                if isinstance(theme, str):
                    return theme.lower().startswith("dark")
        except Exception:
            pass
        return self._is_dark()

    def showEvent(self, event):
        super().showEvent(event)
        # Hook the app theme once the dock is placed under MainWindow, then
        # align immediately (fixes a wrong guess made in __init__).
        if getattr(self, "_theme_hooked", False):
            return
        try:
            main = self._main()
            if main is None:
                return
            main.preferences.sigTreeStateChanged.connect(self._on_app_theme_changed)
            self._theme_hooked = True
            self._on_app_theme_changed()
        except Exception:
            pass

    @pyqtSlot(object, object)
    def _on_app_theme_changed(self, *args):
        """Re-style and rebuild bubbles so baked colors match the app theme."""
        try:
            active, text, thinking = (
                self._stream_active,
                self._stream_buf,
                self._think_buf,
            )
            self._stream_active = False
            self._apply_styles()
            self._render_history()
            if active:
                self._append_stream_start()
                self._stream_buf, self._think_buf = text, thinking
                self._refresh_stream_slot()
        except Exception:
            pass

    def _apply_styles(self):
        dark = self._app_is_dark()
        colors = theme_for(dark)
        hist_bg, card, ink, muted = (
            colors.window,
            colors.surface,
            colors.text,
            colors.muted,
        )
        asst_border, subtle = colors.border, colors.subtle
        self.setStyleSheet(f"""
            #chatHeader {{ background: {card}; border: 1px solid {asst_border}; border-radius: 16px; }}
            #brandIcon {{ font-size: 22px; color: {colors.accent}; }}
            #chatTitle {{ font-size: 15px; font-weight: 800; color: {ink}; letter-spacing: 0.2px; }}
            #chatSubtitle {{ font-size: 11px; color: {muted}; }}
            #providerControl {{ background: {subtle}; border: 1px solid {asst_border}; border-radius: 11px; }}
            #providerDivider {{ background: {asst_border}; border: none; }}
            #providerSelect {{ background: transparent; color: {ink}; border: none; border-radius: 8px;
                               padding: 4px 30px 4px 10px; font-size: 12px; font-weight: 600; }}
            #providerSelect:hover, #providerSelect:on {{ background: {card}; }}
            #providerSelect:focus {{ border: 1px solid {colors.accent}; padding-left: 9px; }}
            #providerSelect:disabled {{ color: {muted}; }}
            #providerSelect::drop-down {{ border: none; width: 26px; }}
            #providerSelect::down-arrow {{ image: none; }}
            #providerSelect QAbstractItemView {{ background: {card}; color: {ink};
                border: 1px solid {asst_border}; border-radius: 10px; padding: 6px;
                outline: none; selection-background-color: {subtle}; selection-color: {ink}; }}
            #providerSelect QAbstractItemView::item {{ min-height: 34px; padding: 2px 10px; border-radius: 6px; }}
            #providerSettingsBtn {{ background: transparent; color: {muted}; border: none;
                                    border-radius: 8px; padding: 4px 12px; font-size: 12px; font-weight: 600; }}
            #providerSettingsBtn:hover {{ background: {card}; color: {colors.accent}; }}
            #providerSettingsBtn:pressed {{ background: {colors.selection}; color: {colors.accent}; }}
            #providerSettingsBtn:focus {{ border: 1px solid {colors.accent}; }}
            #providerSettingsBtn:disabled {{ color: {muted}; }}
            QComboBox {{ background: {card}; color: {ink}; border: 1px solid {asst_border};
                        border-radius: 7px; padding: 5px 24px 5px 8px; font-size: 11px; }}
            QComboBox:hover, QComboBox:focus {{ border-color: {colors.accent}; }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox::down-arrow {{ image: none; }}
            #modePill {{ background: {colors.selection}; color: {colors.accent}; border: 1px solid {asst_border};
                        border-radius: 10px; padding: 3px 10px; font-size: 11px; font-weight: 800; }}
            #chatHistory {{ background: {hist_bg}; border: 1px solid {asst_border}; border-radius: 16px; padding: 10px; font-size: 13px; }}
            #chatHistory QScrollBar:vertical {{ width: 8px; background: transparent; }}
            #chatHistory QScrollBar::handle:vertical {{ background: {asst_border}; border-radius: 4px; }}
            #previewBar {{ background: {card}; border: 1px dashed {asst_border}; border-radius: 12px; }}
            #chatInput {{ border: 1px solid {asst_border}; border-radius: 12px; padding: 8px 12px;
                         background: {card}; color: {ink}; font-size: 13px; }}
            #chatInput:focus {{ border: 1px solid {colors.accent}; }}
            #sendBtn {{ background: {colors.primary};
                       color: white; border: 1px solid {colors.primary}; border-radius: 9px;
                       padding: 8px 20px; font-weight: 700; font-size: 13px; }}
            #sendBtn:hover {{ background: {colors.primary_hover}; border-color: {colors.primary_hover}; }}
            #sendBtn:pressed {{ background: {colors.primary_pressed}; border-color: {colors.primary_pressed}; }}
            #sendBtn:focus {{ border-color: {ink}; }}
            #sendBtn:disabled {{ background: {subtle}; border-color: {asst_border}; color: {muted}; }}
            #stopBtn {{ background: transparent; color: {muted}; border: 1px solid {asst_border};
                       border-radius: 9px; padding: 8px 16px; font-size: 13px; }}
            #stopBtn:hover {{ color: #dc2626; border: 1px solid #dc2626; }}
            #attachBtn {{ background: {subtle}; color: {muted}; border: 1px solid {asst_border};
                         border-radius: 9px; font-size: 15px; }}
            #attachBtn:hover {{ background: {colors.selection}; color: {colors.accent}; border-color: {colors.accent}; }}
            #attachBtn:pressed {{ background: {asst_border}; }}
            #attachBtn:focus {{ border-color: {colors.accent}; }}
            #thumbBtn {{ background: {card}; border: 1px solid {asst_border}; border-radius: 8px; }}
            #thumbBtn:hover {{ border: 1px solid #dc2626; }}
            #applyBtn {{ background: {colors.primary};
                        color: white; border: 1px solid {colors.primary}; border-radius: 9px;
                        padding: 8px 14px; font-weight: 700; font-size: 13px; }}
            #applyBtn:hover {{ background: {colors.primary_hover}; border-color: {colors.primary_hover}; }}
            #applyBtn:pressed {{ background: {colors.primary_pressed}; border-color: {colors.primary_pressed}; }}
            #applyBtn:focus {{ border-color: {ink}; }}
            #applyBtn:disabled {{ background: {subtle}; border-color: {asst_border}; color: {muted}; }}
            #expBtn {{ background: {card}; color: {ink}; border: 1px solid {asst_border};
                      border-radius: 9px; padding: 8px 16px; font-size: 13px; font-weight: 600; }}
            #expBtn:hover {{ background: {colors.selection}; border-color: {colors.accent}; color: {colors.accent}; }}
            #expBtn:pressed {{ background: {subtle}; border-color: {colors.accent}; }}
            #expBtn:focus {{ border-color: {colors.accent}; }}
            #expBtn:disabled, #stopBtn:disabled {{ color: {muted}; background: {subtle}; border-color: {asst_border}; }}
            #iconBtn {{ background: {card}; border: 1px solid {asst_border}; color: {muted}; border-radius: 8px; }}
            #iconBtn:hover {{ background: {colors.selection}; border-color: {colors.accent}; }}
            #iconBtn:pressed {{ background: {subtle}; }}
            #iconBtn:focus {{ border-color: {colors.accent}; }}
            #chatSidebar {{ background: {card}; border: 1px solid {asst_border}; border-radius: 16px; }}
            #sidebarTitle {{ color: {ink}; font-size: 13px; font-weight: 800; }}
            #sessionList {{ background: {card}; color: {ink}; border: none; outline: none; }}
            #sessionList::item {{ background: {subtle}; color: {ink}; border-radius: 10px;
                                 padding: 8px 10px; margin: 3px 0px; font-size: 12px; font-weight: 600; }}
            #sessionList::item:selected {{ background: {colors.selection}; color: {colors.accent}; }}
            #sessionList::item:hover:!selected {{ background: {asst_border}; }}
            #muted {{ color: {muted}; font-size: 12px; }}
            #statusLbl {{ color: {muted}; font-size: 12px; }}
            #statusDot {{ color: #16a34a; font-size: 13px; }}
            QCheckBox {{ color: {muted}; font-size: 12px; spacing: 8px; }}
            QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {asst_border}; background: {card}; }}
            QCheckBox::indicator:hover, QCheckBox::indicator:focus {{ border: 1px solid {colors.accent}; }}
            QCheckBox::indicator:checked {{ background: {colors.primary}; border: 1px solid {colors.primary}; }}
            QCheckBox::indicator:checked:hover {{ background: {colors.primary_hover}; border-color: {colors.primary_hover}; }}
            QCheckBox::indicator:checked:disabled {{ background: {muted}; border-color: {muted}; }}
            """)
        self.provider_combo.setProperty("dark", dark)
        self.effort_combo.setProperty("dark", dark)
        if _HAS_QTA:
            provider_icons = {
                "fireworks": ("fa5s.bolt", "#f59e0b"),
                "featherless": ("fa5s.feather-alt", "#8b5cf6"),
                "anthropic": ("fa5s.asterisk", "#d97757"),
                "custom_openai": ("fa5s.plug", colors.accent),
                "custom_anthropic": ("fa5s.plug", colors.accent),
            }
            for index in range(self.provider_combo.count()):
                name, color = provider_icons[self.provider_combo.itemData(index)]
                self.provider_combo.setItemIcon(index, qta.icon(name, color=color))
            self.api_settings_btn.setIcon(qta.icon("fa5s.sliders-h", color=muted))
            self.attach_btn.setIcon(qta.icon("fa5s.paperclip", color=muted))
            self.send_btn.setIcon(qta.icon("fa5s.paper-plane", color="#ffffff"))
            self.apply_btn.setIcon(qta.icon("fa5s.cube", color="#ffffff"))
            self.undo_btn.setIcon(qta.icon("fa5s.undo", color=muted))
            self.clear_btn.setIcon(qta.icon("fa5s.plus", color=colors.accent))
            self.sidebar_new_btn.setIcon(qta.icon("fa5s.plus", color=colors.accent))
            self.title_icon.setPixmap(
                QPixmap(":/images/icons/cad_studio_mark.png").scaled(
                    26, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        # Reserve the measured width of the longest provider name, including
        # its icon, padding and arrow, rather than clipping it at a fixed width.
        self.provider_combo.ensurePolished()
        self.provider_combo.setMinimumWidth(self.provider_combo.sizeHint().width())

    # ------------- helpers -------------
    def _main(self):
        """Return the MainWindow (walk up - dock() reparents us into a QDockWidget)."""
        w = self.parent()
        seen = set()
        while w is not None and id(w) not in seen:
            seen.add(id(w))
            comps = getattr(w, "components", None)
            if isinstance(comps, dict) and "editor" in comps:
                return w
            w = w.parent() if hasattr(w, "parent") else None
        return None

    # ----- document model -----
    # QTextBrowser loses div backgrounds depending on previously appended
    # content (assistant bubbles rendered without their background after any
    # earlier append). Table-based bubbles survive, and rebuilding the whole
    # document from parts on each change removes append-order quirks entirely.
    def _append_html(self, snippet):
        self._doc_parts.append(snippet)
        from PyQt5.QtGui import QTextCursor

        cursor = self.history.textCursor()
        cursor.movePosition(QTextCursor.End)
        if not self.history.document().isEmpty():
            cursor.insertBlock()
        cursor.insertHtml(snippet)
        self.history.verticalScrollBar().setValue(
            self.history.verticalScrollBar().maximum()
        )

    def _refresh_view(self):
        try:
            self.history.setHtml(
                "<html><body>" + "".join(self._doc_parts) + "</body></html>"
            )
        except Exception:
            pass
        try:
            sb = self.history.verticalScrollBar()
            sb.setValue(sb.maximum())
        except Exception:
            pass

    def _welcome(self):
        """Professional greeting shown on a fresh panel."""
        colors = theme_for(self._app_is_dark())
        self._append_html(
            f"<div style='text-align:center; margin:28px 12px; color:{colors.text};'>"
            "<div style='font-size:21px; font-weight:700;'>From idea to solid.</div>"
            f"<div style='font-size:12px; margin-top:12px; color:{colors.muted};'>"
            "Describe a part or attach a sketch.<br/>"
            "Your assistant builds, checks, and refines it in the 3D view."
            "</div>"
            f"<div style='font-size:12px; margin-top:18px; color:{colors.muted};'>"
            "Try a 30 × 20 mm bracket with four mounting holes."
            "</div></div>"
        )

    def _bubble_table(self, inner, bg, fg, align, width="88%", radius="16px"):
        return (
            "<table width='100%' cellspacing='0' cellpadding='0' "
            "style='margin-top:6px;margin-bottom:6px;'><tr>"
            f"<td align='{align}'>"
            f"<table width='{width}' cellspacing='0' cellpadding='12' "
            f"style='background-color:{bg};color:{fg};border-radius:{radius};"
            f"box-shadow:0 1px 2px rgba(0,0,0,0.04);'>"
            f"<tr><td>{inner}</td></tr></table>"
            "</td></tr></table>"
        )

    def _bubble(self, who, text, kind="assistant", images=()):
        """Render one message as a modern rounded bubble with timestamp."""
        dark = self._app_is_dark()
        colors = theme_for(dark)
        palette = {
            "user": (colors.primary, "#ffffff", "You"),
            "assistant": (colors.surface, colors.text, "Assistant"),
            "system": (colors.window, colors.muted, "Notice"),
            "tool": (colors.subtle, colors.muted, "Tool"),
            "error": (
                ("#3f1d1d", "#fca5a5", "Error")
                if dark
                else ("#fef2f2", "#b91c1c", "Error")
            ),
        }
        bg, fg, label = palette.get(kind, palette["assistant"])
        stamp = self._now()
        imgs = ""
        for b64 in images:
            imgs += (
                f"<img src='data:image/jpeg;base64,{b64}' width='240' "
                f"style='border-radius:8px;margin:2px 4px 2px 0;' />"
            )
        if imgs:
            imgs = f"<div style='margin-bottom:6px;'>{imgs}</div>"
        safe = html.escape(text)
        if kind == "code":
            return self._code_block(text, stamp)
        if kind == "user":
            return self._bubble_table(
                f"{imgs}<div style='line-height:1.45;'>{safe}</div>",
                bg,
                fg,
                "right",
                radius="16px 16px 4px 16px",
            )
        if kind == "system":
            # Vision feedback thumbnails also render here (screenshot the
            # model just saw), so the user can follow what was verified.
            return (
                f"<div style='text-align:center; margin:6px 16px; font-size:11px; "
                f"color:{fg};'>{imgs}{safe}</div>"
            )
        if imgs:
            return self._bubble_table(
                f"{imgs}<div style='line-height:1.45;'>{safe}</div>",
                bg,
                fg,
                "left",
                radius="16px 16px 16px 4px",
            )
        return self._bubble_table(
            f"<div style='line-height:1.45;'>{safe}</div>",
            bg,
            fg,
            "left",
            radius="16px 16px 16px 4px",
        )

    def _code_block(self, code, stamp):
        """Render a generated code block with a working Copy link.

        NOTE: QTextBrowser cannot run JavaScript, so Copy is an anchor
        (cq-copy:<n>) handled by anchorClicked -> clipboard.
        """
        dark = self._app_is_dark()
        colors = theme_for(dark)
        cbg, cfg = colors.subtle, colors.text
        safe = html.escape(code)
        try:
            snippet_id = len(self._code_snippets)
            self._code_snippets.append(code)
        except Exception:
            snippet_id = 0
        header = (
            f"<table width='100%' cellspacing='0' cellpadding='0'><tr>"
            f"<td align='left' style='font-size:11px;color:{colors.muted};'>"
            f"Generated build123d code</td>"
            f"<td align='right'><a href='cq-copy:{snippet_id}' "
            f"style='font-size:11px;color:{colors.accent};text-decoration:none;'>"
            f"[Copy]</a></td></tr></table>"
        )
        body = (
            f"<table width='100%' cellspacing='0' cellpadding='12' "
            f"style='background-color:{cbg};color:{cfg};'>"
            f"<tr><td style='font-family:Consolas,monospace;"
            f"font-size:12px;white-space:pre-wrap;'>"
            f"{safe}</td></tr></table>"
        )
        return (
            f"<div style='margin:8px 2px 12px;'>"
            f"{header}{body}"
            f"<div style='font-size:11px;color:{colors.muted};margin-top:4px;'>"
            f"Applied to the editor &nbsp;·&nbsp; {stamp}</div></div>"
        )

    @pyqtSlot("QUrl")
    def _on_history_anchor(self, url):
        """Handle in-chat links (currently: code Copy buttons)."""
        try:
            if url.scheme() != "cq-copy":
                return
            snippet_id = int(url.path())
            code = self._code_snippets[snippet_id]
        except Exception:
            return
        try:
            from PyQt5.QtWidgets import QApplication

            QApplication.clipboard().setText(code)
            self.status.setText("Code copied to clipboard.")
        except Exception:
            pass

    def _append(self, who, text, images=()):
        kind = {
            "you": "user",
            "assistant": "assistant",
            "system": "system",
            "tool": "tool",
            "tool result": "tool",
            "error": "error",
        }.get(who.lower(), "assistant")
        # User messages with reference images always render the photo bubble.
        if kind == "user" and images:
            self._append_html(self._bubble(who, text[:4000], kind=kind, images=images))
        # Assistant messages containing code get a prose bubble then a code card.
        elif kind == "assistant" and extract_python_code(text):
            self._append_html(self._bubble(who, text, kind="assistant"))
            code = extract_python_code(text)
            self._append_html(self._bubble(who, code, kind="code"))
        elif len(text) > 600 or kind in ("tool", "error"):
            self._append_html(self._bubble(who, text[:4000], kind=kind))
        else:
            self._append_html(self._bubble(who, text, kind=kind, images=images))

    def _set_busy(self, busy, msg=""):
        self.send_btn.setEnabled(not busy)
        self.attach_btn.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.provider_combo.setEnabled(not busy)
        self.api_settings_btn.setEnabled(not busy)
        self.verification.verify_btn.setEnabled(not busy)
        self.verification.checks_btn.setEnabled(not busy)
        self.clear_btn.setEnabled(not busy)
        self.sidebar_new_btn.setEnabled(not busy)
        self.apply_btn.setEnabled(not busy and bool(self.last_code))
        self.undo_btn.setEnabled(not busy and bool(self.code_history))
        self.sidebar.setEnabled(not busy)
        try:
            self.status_dot.setStyleSheet(f"color: {'#f59e0b' if busy else '#16a34a'};")
        except Exception:
            pass
        if busy:
            self._busy_dots = 0
            self._busy_base = msg or "Working"
            if not self._busy_timer.isActive():
                self._busy_timer.start()
            self._tick_busy()
        else:
            if self._busy_timer.isActive():
                self._busy_timer.stop()
            self._save_session()
            self.status.setText(msg or "Ready.")

    def _tick_busy(self):
        """Animated ellipsis on the status line while a request runs."""
        if self.worker is None and not self._operation_active:
            return
        self._busy_dots = (self._busy_dots + 1) % 4
        base = getattr(self, "_busy_base", "Working")
        self.status.setText(f"{base}{'.' * self._busy_dots}")

    def _thinking_block_html(self, thinking_text):
        """Muted reasoning block (shown live during thinking + in transcript)."""
        colors = theme_for(self._app_is_dark())
        fg, edge = colors.muted, colors.border
        body = html.escape(thinking_text).replace("\n", "<br/>")
        return (
            f"<div style='font-size:11px; color:{fg}; border-left:2px solid {edge}; "
            f"padding-left:8px; margin-bottom:6px; line-height:1.45;'>"
            f"<div style='font-weight:700; margin-bottom:2px;'>Thinking…</div>"
            f"<div>{body}</div></div>"
        )

    @staticmethod
    def _thinking_text_of(content):
        """Visible reasoning text from a final message's content blocks."""
        parts = []
        for block in content or []:
            if isinstance(block, dict):
                btype, text = block.get("type"), block.get("thinking", "")
            else:
                btype = getattr(block, "type", "")
                text = getattr(block, "thinking", "") or ""
            if btype == "thinking" and (text or "").strip():
                parts.append(text)
        return "\n".join(parts).strip()

    def _append_thinking(self, thinking_text):
        """Persist a finished reasoning block as its own muted bubble."""
        if (thinking_text or "").strip():
            self._append_html(self._thinking_block_html(thinking_text))

    def _stream_bubble(self, final=False):
        """Live assistant bubble, rebuilt from the buffers on every token."""
        colors = theme_for(self._app_is_dark())
        bg, fg = colors.surface, colors.text
        thinking = ""
        if self._think_buf.strip():
            thinking = self._thinking_block_html(
                self._think_buf[-(20000 if final else 8000) :]
            )
        limit = 100000 if final else 16000
        preview = self._stream_buf[-limit:]
        notice = ""
        if len(self._stream_buf) > limit or len(self._think_buf) > (
            20000 if final else 8000
        ):
            notice = "<div>Showing the latest portion; the full response is kept in the chat session.</div>"
        body = html.escape(preview).replace("\n", "<br/>")
        return self._bubble_table(
            f"<div style='font-size:11px; margin-bottom:4px;'>"
            f"Assistant &gt; {'response' if final else 'live'}</div>{notice}{thinking}"
            f"<div style='line-height:1.45;'>{body}</div>",
            bg,
            fg,
            "left",
            radius="16px 16px 16px 4px",
        )

    def _append_stream_start(self):
        self._stream_buf = ""
        self._think_buf = ""
        self._stream_dirty = False
        self._stream_active = True
        self._doc_parts.append("")  # reserved slot, rewritten per token
        from PyQt5.QtGui import QTextCursor

        cursor = self.history.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertBlock()
        self._stream_start_pos = cursor.position()
        self._refresh_stream_slot()

    def _refresh_stream_slot(self, final=False):
        from PyQt5.QtGui import QTextCursor

        snippet = self._stream_bubble(final=final)
        if self._doc_parts:
            self._doc_parts[-1] = snippet
        else:
            self._doc_parts.append(snippet)
        cursor = self.history.textCursor()
        start = self._stream_start_pos
        if start is None or start > self.history.document().characterCount() - 1:
            cursor.movePosition(QTextCursor.End)
            cursor.insertBlock()
            start = self._stream_start_pos = cursor.position()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.insertHtml(snippet)
        self.history.verticalScrollBar().setValue(
            self.history.verticalScrollBar().maximum()
        )

    @pyqtSlot(str)
    def on_stream_text(self, delta):
        if isinstance(self.sender(), LLMWorker) and self.sender() is not self.worker:
            return
        if not self._stream_active:
            self._append_stream_start()
        self._stream_buf += delta
        self._busy_base = "Writing"
        self._mark_stream_dirty()

    @pyqtSlot(str)
    def on_thinking_text(self, delta):
        if isinstance(self.sender(), LLMWorker) and self.sender() is not self.worker:
            return
        if not self._stream_active:
            self._append_stream_start()
        self._think_buf += delta
        if not self._stream_buf:
            self._busy_base = "Thinking"
        self._mark_stream_dirty()

    def _mark_stream_dirty(self):
        """Schedule a coalesced view refresh (see _stream_timer)."""
        self._stream_dirty = True
        try:
            if not self._stream_timer.isActive():
                self._stream_timer.start()
        except Exception:
            self._refresh_stream_slot()

    @pyqtSlot()
    def _flush_stream_slot(self):
        if not self._stream_active:
            try:
                if self._stream_timer.isActive():
                    self._stream_timer.stop()
            except Exception:
                pass
            return
        if self._stream_dirty:
            self._stream_dirty = False
            self._refresh_stream_slot()

    def _finish_stream_display(self):
        # The bubble is already complete in place; just stop rebuilding it.
        # Flush any trailing deltas first so the persisted bubble is whole.
        try:
            if self._stream_timer.isActive():
                self._stream_timer.stop()
        except Exception:
            pass
        if self._stream_active:
            self._stream_dirty = False
            self._refresh_stream_slot(final=True)
        self._stream_active = False
        self._stream_start_pos = None

    # ------------- reference images -------------
    MAX_IMAGES = 3
    IMAGE_SIDE = 1024  # longest side sent to the model (keeps token cost sane)

    @staticmethod
    def _encode_qimage(image, name):
        """Downscale and JPEG-encode a QImage.

        Returns dict(name, bytes, w, h) or an error string on failure.
        """
        img = QImage(image)
        if img.isNull():
            return f"Could not read image: {name}"
        if max(img.width(), img.height()) > ChatWidget.IMAGE_SIDE:
            img = img.scaled(
                ChatWidget.IMAGE_SIDE,
                ChatWidget.IMAGE_SIDE,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        from PyQt5.QtCore import QBuffer, QIODevice

        buf = QBuffer()
        buf.open(QIODevice.WriteOnly)
        if not img.save(buf, "JPEG", 75):
            return f"Could not encode image: {name}"
        return {
            "name": name,
            "bytes": bytes(buf.data()),
            "w": img.width(),
            "h": img.height(),
        }

    @staticmethod
    def _encode_image(path):
        """Load, downscale and JPEG-encode an image file."""
        return ChatWidget._encode_qimage(QImage(path), Path(path).name)

    def _add_pending_image(self, image, name):
        """Encode one image for the next message.  Returns (added, error)."""
        if len(self.pending_images) >= self.MAX_IMAGES:
            return False, f"Up to {self.MAX_IMAGES} images per message."
        item = self._encode_qimage(image, name)
        if isinstance(item, str):
            return False, item
        self.pending_images.append(item)
        return True, None

    def _clipboard_images(self):
        """Return image candidates from the clipboard, without treating text as an image."""
        mime = QApplication.clipboard().mimeData()
        if mime is None:
            return []

        if mime.hasImage():
            image = mime.imageData()
            if isinstance(image, QPixmap):
                image = image.toImage()
            if isinstance(image, QImage) and not image.isNull():
                return [(image, "Pasted image")]

        # Copying an image file in Explorer produces URLs rather than imageData.
        images = []
        image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        for url in mime.urls() if mime.hasUrls() else []:
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if Path(path).suffix.lower() not in image_suffixes:
                continue
            image = QImage(path)
            if not image.isNull():
                images.append((image, Path(path).name))
        if images:
            return images

        # Some applications expose only a concrete image MIME payload.
        for image_format in mime.formats():
            if not image_format.lower().startswith("image/"):
                continue
            image = QImage.fromData(bytes(mime.data(image_format)))
            if not image.isNull():
                return [(image, "Pasted image")]
        return []

    def _paste_clipboard_images(self):
        """Attach clipboard images and report whether this was an image paste."""
        images = self._clipboard_images()
        if not images:
            return False
        if self.worker is not None:
            self.status.setText(
                "Wait for the current request before attaching an image."
            )
            return True

        added = 0
        error = None
        for image, name in images:
            ok, error = self._add_pending_image(image, name)
            if not ok:
                break
            added += 1
        if added:
            self._refresh_previews()
            noun = "image" if added == 1 else "images"
            self.status.setText(f"Pasted {added} reference {noun}.")
        elif error:
            self.status.setText(error)
        return True

    @pyqtSlot()
    def on_attach(self):
        if self.worker is not None:
            return
        if len(self.pending_images) >= self.MAX_IMAGES:
            self._append("System", f"Up to {self.MAX_IMAGES} images per message.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Attach reference image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        for path in paths:
            if len(self.pending_images) >= self.MAX_IMAGES:
                break
            ok, error = self._add_pending_image(QImage(path), Path(path).name)
            if not ok:
                if error:
                    self._append("Error", error)
                continue
        self._refresh_previews()

    def _refresh_previews(self):
        while self.preview_row.count():
            child = self.preview_row.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
        for idx, item in enumerate(self.pending_images):
            pix = QPixmap()
            pix.loadFromData(item["bytes"])
            thumb = pix.scaled(52, 52, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            btn = QPushButton(self.preview_bar)
            btn.setObjectName("thumbBtn")
            btn.setFixedSize(58, 58)
            btn.setIcon(QIcon(thumb))
            btn.setIconSize(thumb.size())
            btn.setToolTip(f"{item['name']} — click to remove")
            btn.clicked.connect(lambda _=False, i=idx: self._remove_pending(i))
            self.preview_row.addWidget(btn)
        self.preview_row.addStretch(1)
        self.preview_bar.setVisible(bool(self.pending_images))

    @pyqtSlot(int)
    def _remove_pending(self, idx):
        if 0 <= idx < len(self.pending_images):
            self.pending_images.pop(idx)
            self._refresh_previews()

    def _image_block(self, item):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(item["bytes"]).decode("ascii"),
            },
        }

    # ------------- sessions -------------
    # Memory model (Claude Code / Codex style):
    # - each chat is one <id>.json file (messages + design_code + usage)
    #   plus an index.json row (id/title/updated/preview) for the sidebar;
    # - empty chats are LAZY: no file/index row until the first message or
    #   the first applied design, so "+ New chat" never litters the disk;
    # - titles are ChatGPT-style: "New chat" until the first prompt
    #   auto-titles from its opening words (rename anytime via sidebar).
    def _ensure_session(self):
        """Lazily create the active session (so empty chats write nothing)."""
        if self.session_id is None:
            title = (self.session_title or "").strip() or DEFAULT_TITLE
            self.session_title = title
            self.session_id = self._store.create_session(title)
            self._refresh_session_combo()
        return self.session_id

    def _save_session(self):
        if self.session_id is None:
            return  # lazy unsaved chat: nothing on disk yet
        if not self.messages and not (self.session_design or "").strip():
            return  # brand-new untouched session: write nothing yet
        self._store.save_session(
            self.session_id,
            self.messages,
            title=self.session_title,
            model=getattr(self, "_model", ""),
            usage=self.sess_usage,
            design_code=self.session_design,
            requirements=self.design_requirements,
        )
        # Keep the sidebar title/preview in sync (cheap: index rewrite only).
        try:
            self._refresh_session_combo()
        except Exception:
            pass

    def _editor_text(self):
        """Current editor text (the live design), or '' if unavailable."""
        try:
            main = self._main()
            if main is None:
                return ""
            return main.components["editor"].get_text_with_eol() or ""
        except Exception:
            return ""

    def _stash_design(self):
        """Save the live editor text into the active session, then persist."""
        text = self._editor_text()
        if text.strip():
            self.session_design = text
        elif not (self.session_design or "").strip():
            self.session_design = ""
        if self.session_id is None:
            return  # unsaved chat: keep design in memory only
        self._save_session()

    def _refresh_session_combo(self):
        self._loading_combo = True
        try:
            self.session_list.blockSignals(True)
            self.session_list.clear()
            # Pending unsaved chat (ChatGPT-style): shown on top, selected,
            # but creates no file until the first message/design.
            if self.session_id is None:
                pending = QListWidgetItem(f"○ {self.session_title or DEFAULT_TITLE}")
                pending.setData(Qt.UserRole, None)
                pending.setToolTip("Unsaved chat — send a message to keep it.")
                pending.setFlags(pending.flags() & ~Qt.ItemIsEditable)
                self.session_list.addItem(pending)
                pending.setSelected(True)
            for entry in self._store.list_sessions():
                title = entry.get("title") or DEFAULT_TITLE
                item = QListWidgetItem(title)
                item.setData(Qt.UserRole, entry.get("id"))
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                preview = entry.get("preview") or ""
                updated = entry.get("updated") or ""
                tip = title
                if preview:
                    tip += f"\n{preview}"
                if updated:
                    tip += f"\n{updated.replace('T', ' ')}"
                item.setToolTip(tip)
                self.session_list.addItem(item)
                if entry.get("id") == self.session_id:
                    item.setSelected(True)
        finally:
            self.session_list.blockSignals(False)
            self._loading_combo = False

    def _render_history(self, recap=None):
        """Replay stored messages as bubbles (tool noise skipped)."""
        self._doc_parts = []
        self._code_snippets = []
        self.history.clear()
        self._stream_start_pos = None
        if recap:
            self._append("System", recap)
        if not self.messages:
            self._welcome()
            return
        for message in self.messages:
            role, content = message.get("role"), message.get("content")
            if role == "user" and isinstance(content, str):
                if content.strip():
                    self._append("You", content)
            elif role == "user" and isinstance(content, list):
                texts, thumbs = [], []
                shot_thumbs = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and block.get("text", "").strip():
                        texts.append(block["text"])
                    elif block.get("type") == "image":
                        data = (block.get("source") or {}).get("data") or ""
                        if data:
                            thumbs.append(data)
                    elif block.get("type") == "tool_result":
                        inner = block.get("content", "")
                        if isinstance(inner, list):
                            for sub in inner:
                                if not isinstance(sub, dict):
                                    continue
                                if sub.get("type") == "image":
                                    data = (sub.get("source") or {}).get("data") or ""
                                    if data:
                                        shot_thumbs.append(data)
                if texts or thumbs:
                    self._append(
                        "You", "\n".join(texts) or "(reference images)", images=thumbs
                    )
                if shot_thumbs:
                    # Screenshots the model saw (auto visual feedback).
                    self._append(
                        "System",
                        "Screenshot sent to the model for visual check.",
                        images=shot_thumbs,
                    )
            elif role == "assistant" and isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict):
                        btype, text = block.get("type"), block.get("text", "")
                    else:
                        btype, text = getattr(block, "type", ""), getattr(
                            block, "text", ""
                        )
                    if btype == "text" and (text or "").strip():
                        texts.append(text)
                shown = "\n".join(texts).strip()
                if shown:
                    self._append("Assistant", shown)
        sb = self.history.verticalScrollBar()
        sb.setValue(sb.maximum())

    @pyqtSlot()
    def on_clear(self):
        self.on_new_session()

    @pyqtSlot()
    def on_new_session(self):
        """Start a fresh chat with an EMPTY design (nothing auto-drawn).

        The previous chat keeps its transcript + design on disk (see
        _stash_design). The new chat stays lazy/unsaved — no session file
        until the first message or applied design — with an empty editor
        and an empty 3D board. Nothing is rendered automatically.
        """
        if self.worker is not None or self._operation_active:
            self.status.setText("Stop the running request first.")
            return
        self._stash_design()
        # Lazy ChatGPT-style chat: no file/index row yet.
        self.session_id = None
        self.session_title = DEFAULT_TITLE
        self.messages = []
        self.last_code = ""
        self.session_design = ""
        self._set_requirements([])
        self.sess_usage = {"in": 0, "out": 0}
        self.last_usage = {"in": 0, "out": 0}
        self._turn_steps = 0
        self._update_meter(0)
        self.apply_btn.setEnabled(False)
        self.pending_images = []
        self._refresh_previews()
        self._refresh_session_combo()
        self._render_history()
        self._reset_editor_to_empty()
        self._clear_board()
        self._append("System", "New chat started — empty design, nothing rendered yet.")
        self._set_busy(False, "New chat started.")

    @pyqtSlot(QListWidgetItem)
    def on_session_list_clicked(self, item):
        if self._loading_combo or item is None:
            return
        session_id = item.data(Qt.UserRole)
        if session_id == self.session_id:
            return
        if not session_id:
            # Pending "○ New chat" row: start a fresh unsaved chat instead.
            if self.session_id is None:
                return
            if self.worker is not None:
                self.status.setText("Stop the running request before switching chats.")
                self._refresh_session_combo()
                return
            self._stash_design()
            self.session_id = None
            self.session_title = DEFAULT_TITLE
            self.messages = []
            self.session_design = ""
            self.last_code = ""
            self.apply_btn.setEnabled(False)
            self.pending_images = []
            self._refresh_previews()
            self.sess_usage = {"in": 0, "out": 0}
            self.last_usage = {"in": 0, "out": 0}
            self._turn_steps = 0
            self._render_history()
            self._update_meter(0)
            self._reset_editor_to_empty()
            self._clear_board()
            self._append(
                "System", "New chat started — empty design, nothing rendered yet."
            )
            self._set_busy(False, "New chat started.")
            self._refresh_session_combo()
            return
        if self.worker is not None:
            self.status.setText("Stop the running request before switching chats.")
            self._refresh_session_combo()  # snap the list back
            return
        self._stash_design()
        messages, meta = self._store.load_session(session_id)
        if messages is None:
            self._append("Error", "Could not load that chat.")
            self._refresh_session_combo()
            return
        self.session_id = session_id
        self.session_title = (meta or {}).get("title") or DEFAULT_TITLE
        self.messages = messages
        self.session_design = (meta or {}).get("design_code", "") or ""
        self.last_code = ""
        self.apply_btn.setEnabled(False)
        self.pending_images = []
        self._refresh_previews()
        usage = (meta or {}).get("usage") or {}
        self.sess_usage = {
            "in": int(usage.get("in", 0) or 0),
            "out": int(usage.get("out", 0) or 0),
        }
        self.last_usage = {"in": 0, "out": 0}
        self._turn_steps = 0
        recap = (
            f"Resumed “{self.session_title}” — {len(messages)} messages · "
            f"in/out {self._fmt_tokens(self.sess_usage['in'])}"
            f"/{self._fmt_tokens(self.sess_usage['out'])} total"
        )
        self._render_history(recap=recap)
        self._update_meter()
        design_note = self._restore_session_design()
        self._append("System", design_note)
        self._set_busy(False, f"Switched to “{self.session_title}”.")
        self._refresh_session_combo()

    def _session_context_menu(self, pos):
        item = self.session_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        rename_act = menu.addAction("Rename")
        delete_act = menu.addAction("Delete")
        action = menu.exec_(self.session_list.mapToGlobal(pos))
        if action == rename_act:
            self.session_list.editItem(item)
        elif action == delete_act:
            self._delete_current_session()

    @pyqtSlot(QListWidgetItem)
    def on_session_item_changed(self, item):
        """Commit an edited session list item as the chat title."""
        if self._loading_combo or item is None:
            return
        session_id = item.data(Qt.UserRole)
        if session_id is None or session_id != self.session_id:
            self._refresh_session_combo()  # never rename the unsaved row
            return
        new_title = " ".join(item.text().split())
        if not new_title or new_title == self.session_title:
            self._refresh_session_combo()  # revert empty / unchanged edits
            return
        if self._store.rename_session(self.session_id, new_title):
            self.session_title = new_title[:80]
            self.status.setText(f"Renamed to “{self.session_title}”.")
        self._refresh_session_combo()

    def _delete_current_session(self):
        if self.worker is not None:
            self.status.setText("Stop the running request first.")
            return
        if self.session_id is None:
            self.on_new_session()
            return
        title = self.session_title
        if (
            QMessageBox.question(
                self,
                "Delete chat",
                f"Delete “{title}” permanently?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        self._store.delete_session(self.session_id)
        remaining = self._store.list_sessions()
        if remaining:
            self.session_id = remaining[0]["id"]
            messages, meta = self._store.load_session(self.session_id)
            self.messages = messages or []
            self.session_title = (meta or {}).get("title") or DEFAULT_TITLE
            self.session_design = (meta or {}).get("design_code", "") or ""
        else:
            # Nothing left: fall back to a lazy unsaved chat (no junk file).
            self.session_id = None
            self.session_title = DEFAULT_TITLE
            self.messages = []
            self.session_design = ""
        self.last_code = ""
        self.apply_btn.setEnabled(False)
        self.pending_images = []
        self._refresh_previews()
        self._refresh_session_combo()
        self._render_history()
        design_note = self._restore_session_design()
        self._append("System", design_note)
        self._set_busy(False, "Ready.")

    # ------------- Claude/Codex UX: modes, slash commands, checkpoints -------------
    @staticmethod
    def _fmt_tokens(n):
        try:
            n = int(n)
        except Exception:
            return "?"
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    def _set_plan(self, on):
        self.plan_mode = bool(on)
        try:
            self.plan_check.blockSignals(True)
            self.plan_check.setChecked(self.plan_mode)
        finally:
            self.plan_check.blockSignals(False)
        try:
            self.mode_pill.setVisible(self.plan_mode)
        except Exception:
            pass
        self._append(
            "System",
            (
                "Plan mode ON: code is validated but never auto-applied. "
                "Review it, then press “Insert code + Render”. (/plan to toggle)"
                if self.plan_mode
                else "Plan mode OFF: generated code auto-applies again."
            ),
        )

    @pyqtSlot(bool)
    def on_plan_toggled(self, on):
        if bool(on) != self.plan_mode:
            self._set_plan(on)

    # ----- reasoning effort (OpenCode/Codex style: low/medium/high/max) -----
    @staticmethod
    def _effort_tooltip():
        lines = ["Reasoning effort — thinking budget + output ceiling per turn:"]
        for name in ("low", "medium", "high", "max"):
            spec = EFFORTS[name]
            budget = spec["budget"]
            lines.append(
                f"· {name.title()}: "
                f"{'thinking off' if budget is None else f'think ~{budget} tokens'}"
                f" · max {spec['max_tokens']} tokens"
            )
        lines.append("Thinking applies to reasoning models (glm via Fireworks).")
        return "\n".join(lines)

    @staticmethod
    def _effort_summary():
        lines = ["Reasoning effort — thinking budget + output ceiling per turn:"]
        for name in ("low", "medium", "high", "max"):
            spec = EFFORTS[name]
            budget = spec["budget"]
            lines.append(
                f"  {name:<7} {'thinking off' if budget is None else f'think ~{budget} tokens'}"
                f" · max {spec['max_tokens']} tokens"
            )
        lines.append("Usage: /effort [low|medium|high|max]")
        return "\n".join(lines)

    def _set_effort(self, level, announce=False):
        """Set the effort level and sync the combo + model label."""
        level = normalize_effort(level)
        self.effort = level
        try:
            self.effort_combo.blockSignals(True)
            self.effort_combo.setCurrentText(level.title())
        finally:
            self.effort_combo.blockSignals(False)
        self._refresh_model_lbl()
        if announce:
            self._append(
                "System", f"Effort set to {level.title()}.\n{self._effort_summary()}"
            )

    @pyqtSlot(str)
    def on_effort_changed(self, text):
        level = normalize_effort(text)
        if level != self.effort:
            self._set_effort(level)
            self.status.setText(
                f"Effort: {level.title()} "
                f"(think ~{EFFORTS[level]['budget'] or 'off'} · "
                f"max {EFFORTS[level]['max_tokens']})."
            )

    def _selected_provider(self):
        return self.provider_combo.currentData() or "fireworks"

    def _selected_config(self):
        return self._provider_settings.get(self._selected_provider())

    def _on_provider_changed(self, _index):
        provider = self._selected_provider()
        self._provider_settings.select(provider)
        self._client = None
        self._model = ""
        self._refresh_provider_pill()
        self._refresh_model_lbl()
        self.status.setText(
            f"Selected {PROVIDERS[provider].label}. Configure it in Provider settings if needed."
        )

    def _edit_provider_settings(self):
        dialog = ProviderSettingsDialog(self._selected_config(), self)
        if dialog.exec_() == dialog.Accepted:
            self._provider_settings.save(dialog.config())
            self._client = None
            self._model = ""
            self._refresh_provider_controls()
            self._refresh_model_lbl()
            self.status.setText("API settings saved. Ready for the next message.")

    def _refresh_model_lbl(self):
        try:
            model = (
                getattr(self, "_model", "")
                or self._selected_config().model
                or "not configured"
            )
            self.model_lbl.setText(f"Model: {model} · Effort: {self.effort.title()}")
            self.model_lbl.setToolTip(
                f"Selected provider: {PROVIDERS[self._selected_provider()].label}"
            )
        except Exception:
            pass

    def _cmd_effort(self, text):
        """'/effort' shows the table; '/effort high' switches level."""
        parts = (text or "").strip().split()
        if len(parts) < 2:
            self._append(
                "System",
                f"Current effort: {self.effort.title()} "
                f"(model: {getattr(self, '_model', '') or 'from .env'})\n"
                + self._effort_summary(),
            )
            return
        level = normalize_effort(parts[1])
        if parts[1].strip().lower() not in EFFORTS:
            self._append(
                "System",
                f"Unknown effort “{parts[1]}”. Valid: low, medium, high, max.\n"
                + self._effort_summary(),
            )
            return
        self._set_effort(level, announce=True)

    # ----- slash commands -----
    @pyqtSlot()
    def _update_slash_popup(self):
        """Show command completion only for a bare '/<letters>' composer."""
        try:
            text = self.input.toPlainText()
        except Exception:
            return
        if text.startswith("/") and "\n" not in text and " " not in text:
            self.slash_completer.setCompletionPrefix(text[1:])
            cr = self.input.cursorRect()
            cr.setWidth(
                self.slash_completer.popup().sizeHintForColumn(0)
                + self.slash_completer.popup().verticalScrollBar().sizeHint().width()
            )
            self.slash_completer.complete(cr)
        else:
            self.slash_completer.popup().hide()

    @pyqtSlot(str)
    def _accept_slash(self, choice):
        """Picker selection runs immediately (Codex-style fast path)."""
        cmd = choice.split()[0] if choice else ""
        if cmd in SLASH_COMMANDS:
            self.input.clear()
            self._run_slash(cmd)

    def _run_slash(self, text):
        """Dispatch a '/command' line. Returns True if it was a command."""
        cmd = text.strip().split()[0].lower() if text.strip() else ""
        if not cmd.startswith("/"):
            return False
        handler = {
            "/new": lambda: self.on_new_session(),
            "/clear": lambda: self.on_new_session(),
            "/status": self._cmd_status,
            "/usage": self._cmd_usage,
            "/plan": lambda: self._set_plan(not self.plan_mode),
            "/undo": self.on_undo,
            "/export": self._cmd_export,
            "/examples": lambda: self._append("System", self.get_examples()),
            "/skills": lambda: self._cmd_skills(text),
            "/compact": self._cmd_compact,
            "/model": self._cmd_model,
            "/effort": lambda: self._cmd_effort(text),
            "/help": lambda: self._append("System", self._help_text()),
        }.get(cmd)
        if handler is None:
            self._append(
                "System",
                f"Unknown command “{cmd}”. Type /help for the list.",
            )
            return True
        try:
            handler()
        except Exception as e:
            self._append("Error", f"Command failed: {e}")
        return True

    @staticmethod
    def _help_text():
        lines = ["Slash commands — type / to complete, Enter to run:"]
        lines += [f"  {cmd:<10} {desc}" for cmd, desc in SLASH_COMMANDS.items()]
        return "\n".join(lines)

    def _cmd_skills(self, text):
        arguments = text.split()[1:]
        output = (
            self._skills.view(arguments[0], draft="draft" in arguments[1:])
            if arguments
            else self._skills.summary()
        )
        self._append("System", output)

    def _payload_chars(self):
        try:
            from ..ai.cad_agent import compact_messages

            import json

            return len(json.dumps(compact_messages(self.messages), default=str))
        except Exception:
            return 0

    def _mode_name(self):
        if self.plan_mode:
            return "Plan (validate only)"
        return "Auto-apply" if self.preferences["Auto-apply code"] else "Manual"

    def _cmd_status(self):
        config = self._selected_config()
        model = getattr(self, "_model", "") or config.model or "not configured"
        spec = EFFORTS[self.effort]
        budget = spec["budget"]
        self._append(
            "System",
            "Session status\n"
            f"· Provider: {PROVIDERS[config.provider].label}\n"
            f"· Model: {model}\n"
            f"· Effort: {self.effort.title()} "
            f"({'thinking off' if budget is None else f'think ~{budget}'}"
            f" · max {spec['max_tokens']}) (/effort to change)\n"
            f"· Mode: {self._mode_name()} · Self-fix: {'on' if self.autofix.isChecked() else 'off'}"
            f" · Vision: {'on' if self._vision_enabled() else 'off'}\n"
            f"· Chat: “{self.session_title}” ({len(self.messages)} messages)\n"
            f"· Context payload: ~{self._fmt_tokens(self._payload_chars() // 4)} tokens",
        )

    def _cmd_usage(self):
        self._append(
            "System",
            "Token usage — this chat\n"
            f"· Session total in/out: {self._fmt_tokens(self.sess_usage['in'])} / "
            f"{self._fmt_tokens(self.sess_usage['out'])}\n"
            f"· Last turn in/out: {self._fmt_tokens(self.last_usage['in'])} / "
            f"{self._fmt_tokens(self.last_usage['out'])}",
        )

    def _cmd_model(self):
        config = self._selected_config()
        model = getattr(self, "_model", "") or config.model or "not configured"
        self._append(
            "System",
            f"Provider: {PROVIDERS[config.provider].label}\n"
            f"Selected model: {model}\n"
            "Use Provider settings to change its API key, model, or base URL.",
        )

    def _cmd_compact(self):
        import json

        try:
            from ..ai.cad_agent import compact_messages

            raw = len(json.dumps(self.messages, default=str))
            packed = len(json.dumps(compact_messages(self.messages), default=str))
            saved = 100 * (1 - packed / raw) if raw else 0
            self._append(
                "System",
                "Auto-compaction (s08-style, runs before every call)\n"
                f"· Stored: {raw} chars → sent: {packed} chars ({saved:.0f}% smaller)\n"
                "· Stale thinking dropped · old code/results shortened · "
                "latest turn always verbatim",
            )
        except Exception as e:
            self._append("Error", f"Could not measure context: {e}")

    def _transcript_markdown(self):
        out = [f"# {self.session_title}", ""]
        for message in self.messages:
            role, content = message.get("role"), message.get("content")
            if role == "user" and isinstance(content, str):
                out += ["## You", "", content, ""]
            elif role == "user" and isinstance(content, list):
                parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                imgs = sum(
                    1
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "image"
                )
                shots = 0
                for b in content:
                    if not isinstance(b, dict) or b.get("type") != "tool_result":
                        continue
                    inner = b.get("content", "")
                    if isinstance(inner, list):
                        shots += sum(
                            1
                            for s in inner
                            if isinstance(s, dict) and s.get("type") == "image"
                        )
                if imgs:
                    parts.append(f"_[{imgs} reference image(s) attached]_")
                if shots:
                    parts.append(f"_[{shots} 3D screenshot(s) sent for visual check]_")
                out += ["## You", "", "\n".join(parts), ""]
            elif role == "assistant" and isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict):
                        btype, text = block.get("type"), block.get("text", "")
                    else:
                        btype = getattr(block, "type", "")
                        text = getattr(block, "text", "") or ""
                    if btype == "text" and text.strip():
                        texts.append(text)
                    elif btype == "tool_use":
                        name = (
                            block.get("name") if isinstance(block, dict) else block.name
                        )
                        texts.append(f"_⚙ tool: {name}_")
                if texts:
                    out += ["## Assistant", "", "\n".join(texts), ""]
        return "\n".join(out).strip() + "\n"

    def _cmd_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save chat transcript",
            f"{self.session_title}.md",
            "Markdown (*.md)",
        )
        if not path:
            return
        try:
            Path(path).write_text(self._transcript_markdown(), encoding="utf-8")
            self._append("System", f"Transcript saved to {path}")
        except Exception as e:
            self._append("Error", f"Export failed: {e}")

    # ----- checkpoints / undo -----
    @staticmethod
    def _diffstat(old, new):
        import difflib

        try:
            ins = dels = 0
            for line in difflib.unified_diff(
                old.splitlines(), new.splitlines(), lineterm=""
            ):
                if line.startswith("+++") or line.startswith("---"):
                    continue
                if line.startswith("+"):
                    ins += 1
                elif line.startswith("-"):
                    dels += 1
            return f"+{ins}/−{dels} lines"
        except Exception:
            return ""

    def _push_checkpoint(self, code):
        if not code:
            return
        if self.code_history and self.code_history[-1]["code"] == code:
            return
        self.code_history.append({"time": self._now(), "code": code})
        self.code_history = self.code_history[-20:]
        try:
            self.undo_btn.setEnabled(True)
        except Exception:
            pass

    @pyqtSlot()
    def on_undo(self):
        """Restore the editor to the code before the last apply (rewind)."""
        if not self.code_history:
            self._append("System", "Nothing to undo — no earlier editor state saved.")
            return
        prev = self.code_history.pop()["code"]
        try:
            main = self._main()
            if main is None:
                self._append("Error", "Main window not found.")
                return
            main.components["editor"].set_text(prev)
            try:
                main.components["debugger"].render()
            except Exception:
                pass
            self.last_code = prev
            self._append("System", "Undone — editor restored to the previous version.")
        except Exception as e:
            self._append("Error", f"Undo failed: {e}")
        finally:
            try:
                self.undo_btn.setEnabled(bool(self.code_history))
            except Exception:
                pass

    # ----- usage meter (Claude-Code status-line pattern) -----
    def _track_usage(self, resp):
        try:
            usage = getattr(resp, "usage", None)
            inp = int(getattr(usage, "input_tokens", 0) or 0)
            out = int(getattr(usage, "output_tokens", 0) or 0)
        except Exception:
            return
        self.last_usage = {"in": inp, "out": out}
        self.sess_usage["in"] += inp
        self.sess_usage["out"] += out
        self._update_meter()

    def _update_meter(self, payload_chars=None):
        # ASCII-only: the ↑↓Σ glyphs don't render in some UI fonts and showed
        # up as mystery numbers (see screenshot). Words always render.
        try:
            if payload_chars is None:
                payload_chars = self._payload_chars()
            ctx = self._fmt_tokens(payload_chars // 4)
            last = (
                f"{self._fmt_tokens(self.last_usage['in'])}"
                f"/{self._fmt_tokens(self.last_usage['out'])} last"
            )
            sess = (
                f"{self._fmt_tokens(self.sess_usage['in'])}"
                f"/{self._fmt_tokens(self.sess_usage['out'])} total"
            )
            self.meter_lbl.setText(f"ctx ~{ctx} in/out · {last} · {sess}")
        except Exception:
            pass

    @staticmethod
    def _tool_arg_preview(name, args):
        """One-line arg summary so tool calls stay single-line in the timeline."""
        try:
            code = args.get("code")
            if isinstance(code, str) and code:
                lines = len(code.splitlines())
                return f"({lines} lines, {len(code)} chars)"
            short = str(args)
            return f"({short[:80]})" if len(short) > 2 else ""
        except Exception:
            return ""

    def _append_step(self, text, ok=None):
        """One-line tool-timeline entry (Focus-view style: quiet on success)."""
        mark = ""
        if ok is True:
            mark = "<b style='color:#16a34a;'>[ok]</b> "
        elif ok is False:
            mark = "<b style='color:#dc2626;'>[fail]</b> "
        self._append_html(
            f"<div style='margin:1px 2px 1px 8px; font-size:11.5px; "
            f"font-family:Consolas,monospace;'>{mark}{html.escape(text)}</div>"
        )

    # ------------- send / loop -------------
    @pyqtSlot()
    def on_send(self):
        prompt = self.input.toPlainText().strip()
        if (
            (not prompt and not self.pending_images)
            or self.worker is not None
            or self._operation_active
        ):
            return
        if sum(w.isRunning() for w in self._active_workers) >= 2:
            self.status.setText(
                "Two cancelled requests are still disconnecting. Please wait for their network timeouts."
            )
            return
        # Slash commands run locally, never reach the model.
        if prompt.startswith("/") and not self.pending_images:
            self.input.clear()
            self._hist_idx = None
            self._run_slash(prompt)
            return
        try:
            client, model = build_client(config=self._selected_config())
        except RuntimeError as e:
            self._append("System", str(e))
            self.status.setText(str(e))
            return
        self._client, self._model = client, model
        self._requirements_recorded_this_turn = False
        self.verification.invalidate()
        self.verification_report = None
        self._verification_code = None
        self._visual_capture_revision = None
        self._validated_job = None
        self._refresh_provider_pill()
        self._refresh_model_lbl()
        self._ensure_session()
        _has_images = bool(self.pending_images)
        if is_auto_title(self.session_title) and (prompt or _has_images):
            if _has_images and not prompt:
                blocks_for_title = [
                    {"type": "image"},
                    {"type": "text", "text": "Reference images"},
                ]
                self.session_title = autotitle(
                    [{"role": "user", "content": blocks_for_title}]
                )
            elif prompt:
                self.session_title = autotitle([{"role": "user", "content": prompt}])
            self._save_session()
            self._refresh_session_combo()
        images = list(self.pending_images)
        self.pending_images = []
        self._refresh_previews()
        if images:
            blocks = [self._image_block(item) for item in images]
            blocks.append(
                {
                    "type": "text",
                    "text": prompt
                    or "Model the part shown in the attached reference image(s) in mm.",
                }
            )
            self.messages.append({"role": "user", "content": blocks})
            thumbs = [
                base64.b64encode(item["bytes"]).decode("ascii") for item in images
            ]
            self._append("You", prompt or "(reference images)", images=thumbs)
        else:
            self.messages.append({"role": "user", "content": prompt})
            self._append("You", prompt)
        self.input.clear()
        self._hist_idx = None
        if prompt and (not self.sent_history or self.sent_history[-1] != prompt):
            self.sent_history.append(prompt)
            self.sent_history = self.sent_history[-100:]
        self.iters_left = int(self.preferences["Max tool iterations"])
        self._turn_steps = 0
        self._finalizing = False
        self._stop_requested = False
        self._request_rendered = False
        self._kick_llm()

    def _refresh_provider_controls(self):
        provider = self.provider_combo.currentData()
        self.provider_combo.setToolTip(f"AI provider: {PROVIDERS[provider].label}")
        self.api_settings_btn.setToolTip(
            f"Manage {PROVIDERS[provider].label} API key and model"
        )

    def _refresh_provider_pill(self):
        self._refresh_provider_controls()

    def eventFilter(self, obj, event):
        """Enter sends, Shift+Enter inserts a newline, Up/Down recalls history."""
        if obj is self.input and event.type() == event.KeyPress:
            key = event.key()
            mods = event.modifiers()
            if event.matches(QKeySequence.Paste) and self._paste_clipboard_images():
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter) and not (mods & Qt.ShiftModifier):
                try:
                    if self.slash_completer.popup().isVisible():
                        return False  # let the completer accept the command
                except Exception:
                    pass
                self.on_send()
                return True
            if key in (Qt.Key_Up, Qt.Key_Down) and self.sent_history:
                try:
                    if self.slash_completer.popup().isVisible():
                        return False
                except Exception:
                    pass
                cursor = self.input.textCursor()
                first = cursor.blockNumber() == 0
                last = cursor.blockNumber() == self.input.blockCount() - 1
                if (key == Qt.Key_Up and first) or (key == Qt.Key_Down and last):
                    self._recall_history(-1 if key == Qt.Key_Up else 1)
                    return True
        return super().eventFilter(obj, event)

    def _recall_history(self, direction):
        """Step through sent prompts like a terminal (Claude/Codex Up/Down)."""
        if not self.sent_history:
            return
        if self._hist_idx is None:
            self._hist_idx = len(self.sent_history) if direction < 0 else -1
        self._hist_idx = max(
            -1, min(len(self.sent_history) - 1, self._hist_idx + direction)
        )
        text = "" if self._hist_idx < 0 else self.sent_history[self._hist_idx]
        try:
            self.input.blockSignals(True)
            self.input.setPlainText(text)
            cursor = self.input.textCursor()
            cursor.movePosition(cursor.End)
            self.input.setTextCursor(cursor)
        finally:
            self.input.blockSignals(False)

    @staticmethod
    def _now():
        return datetime.now().strftime("%H:%M")

    @pyqtSlot()
    def on_stop(self):
        """Cancel the in-flight request and halt the agent loop."""
        self.iters_left = 0
        self._stop_requested = True
        main = self._main()
        if main is not None:
            main.components["debugger"].cancel_render()
        if self._operation_active:
            self.status.setText("Stopping CAD operation…")
            return
        w = self.worker
        if w is not None:
            w.requestInterruption()
            # Discard late replies while the socket finishes in the background.
            # Keep a strong reference until QThread.finished to prevent teardown crashes.
            self.worker = None
            self._finish_stream_display()
            self._set_busy(
                False,
                "Stopped. The network request is disconnecting in the background.",
            )
            return
        self._finish_stream_display()
        self._append("System", "Stopped by user.")
        self._set_busy(False)

    def _kick_llm(self):
        self._set_busy(
            True, "Preparing final design" if self._finalizing else "Working"
        )
        # s08-style: send a compacted copy (stale thinking dropped, old code
        # and old tool results shortened). self.messages stays complete.
        from ..ai.cad_agent import thinking_params_for

        payload = self.messages
        thinking = thinking_params_for(self._model, self.effort)
        max_tokens = max_tokens_for_effort(self.effort)
        learning = (
            bool(self.preferences["Learn reusable CAD skills"]) and not self.plan_mode
        )
        tools = (
            []
            if self._finalizing
            else [tool for tool in TOOLS if learning or tool["name"] != "skill_manage"]
        )
        self._stream_buf = ""
        self._think_buf = ""
        if self.stream_check.isChecked():
            self._append_stream_start()
            self.worker = StreamWorker(
                self._client,
                self._model,
                payload,
                self,
                thinking=thinking,
                max_tokens=max_tokens,
                tools=tools,
                skills=self._skills,
                learning=learning,
            )
            self.worker.text_delta.connect(self.on_stream_text)
            self.worker.thinking_delta.connect(self.on_thinking_text)
            self.worker.stream_reset.connect(self._reset_live_reply)
        else:
            self.worker = LLMWorker(
                self._client,
                self._model,
                payload,
                self,
                thinking=thinking,
                max_tokens=max_tokens,
                tools=tools,
                skills=self._skills,
                learning=learning,
            )
        self.worker.finished_ok.connect(self.on_llm_ok)
        self.worker.finished_err.connect(self.on_llm_err)
        self.worker.status_changed.connect(self._recovery_status)
        self._retain_worker(self.worker)
        self.worker.start()

    def _retain_worker(self, worker):
        """Keep threads alive until run() has actually returned."""
        self._active_workers.append(worker)

        def finished():
            if worker in self._active_workers:
                self._active_workers.remove(worker)
            worker.deleteLater()

        worker.finished.connect(finished)

    def _continue_tool_loop(self):
        """Allow one final reply after the bounded tool rounds are exhausted."""
        if self._stop_requested:
            self._set_busy(False, "Stopped by user.")
            return
        if self.iters_left > 0 and not self._finalizing:
            self._kick_llm()
            return
        if self._finalizing:
            self._set_busy(
                False,
                "Automatic attempts finished. Review the last result before retrying.",
            )
            return
        self._finalizing = True
        if self._request_rendered:
            instruction = (
                "The model was rendered successfully and the tool-round limit has been reached. "
                "Summarize the actual result and any unresolved issues. Do not call more tools "
                "or repeat the rendered script. Do not claim checks that did not happen."
            )
        else:
            instruction = (
                "The tool-round limit has been reached. Do not call any more tools. "
                "Using the original request and tool results, return ONE complete ```python "
                "script for the finished design, ending in show_object(result). The app will "
                "validate it before applying it. Preserve all requested dimensions and features. "
                "If you cannot finish, explain the unresolved problem instead of claiming success."
            )
        self.messages.append({"role": "user", "content": instruction})
        self._append("System", "Preparing a final response from the tool results.")
        self._kick_llm()

    @staticmethod
    def _tool_failed(output):
        return output.lstrip().startswith(
            (
                "Error:",
                "Traceback",
                "CRASH:",
                "TIMEOUT:",
                "Rendered WITH traceback:",
                "Subagent failed:",
                "Error reading editor:",
                "Error listing examples:",
                "Error applying code:",
                "Error exporting:",
            )
        )

    @pyqtSlot()
    def _reset_live_reply(self):
        if isinstance(self.sender(), LLMWorker) and self.sender() is not self.worker:
            return
        self._stream_buf = ""
        self._think_buf = ""
        if self._stream_active:
            self._stream_dirty = True

    @pyqtSlot(str)
    def _recovery_status(self, message):
        if isinstance(self.sender(), LLMWorker) and self.sender() is not self.worker:
            return
        self._busy_base = message
        self.status.setText(message)

    @pyqtSlot(object)
    def on_llm_ok(self, resp):
        if isinstance(self.sender(), LLMWorker) and self.sender() is not self.worker:
            return
        self.worker = None
        self._finish_stream_display()
        content = getattr(resp, "content", [])
        stop = getattr(resp, "stop_reason", "") or ""
        if self._stop_requested or (self.iters_left <= 0 and not self._finalizing):
            self._set_busy(False, "Stopped. No further code was applied.")
            return
        if stop in ("max_tokens", "length"):
            self.on_llm_err(
                "Reply is incomplete after automatic recovery. No partial code was applied."
            )
            return
        self._track_usage(resp)
        self.messages.append({"role": "assistant", "content": content})
        tool_calls = [b for b in content if getattr(b, "type", "") == "tool_use"]
        if self._finalizing and tool_calls:
            # Close the tool pairs without dispatching an out-of-budget call.
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Tool-round limit reached. This tool was not executed.",
                            "is_error": True,
                        }
                        for block in tool_calls
                    ],
                }
            )
            self._set_busy(
                False,
                "The model requested more tools after the final attempt. Simplify the request or increase Max tool iterations in Preferences.",
            )
            return
        texts = [b.text for b in content if getattr(b, "type", "") == "text"]
        shown = " ".join(texts).strip()
        thinking_text = self._thinking_text_of(content)
        streamed_thinking = bool(self._think_buf.strip())
        self._think_buf = ""
        if not shown and not tool_calls and thinking_text and not streamed_thinking:
            # Thinking-only reply on the non-streaming path: show the
            # reasoning instead of a "no text" error.
            self._append_thinking(thinking_text)
            self._set_busy(False, "Ready (reasoning shown, no reply text).")
            return
        if not shown and not tool_calls:
            # Empty reply: thinking-only or truncated (stop=max_tokens).
            # Drop the empty assistant turn so a retry starts clean.
            try:
                if self.messages and self.messages[-1].get("role") == "assistant":
                    self.messages.pop()
            except Exception:
                pass
            if stop == "max_tokens":
                self._append(
                    "System",
                    "Reply was cut off (token limit) before any code was produced. "
                    "Please try a shorter request, or split it: e.g. vase body first, "
                    "then details — and press Send again.",
                )
                self.status.setText(
                    "Reply truncated (max_tokens). Try again / split request."
                )
            else:
                self._append(
                    "System",
                    (
                        "The model returned no text. Press Send again to retry."
                        if not streamed_thinking
                        else "Reasoning shown above, but no reply text followed. "
                        "Press Send again to retry."
                    ),
                )
                self.status.setText("Empty reply — retry.")
            self._set_busy(False)
            return
        streamed = bool(self._stream_buf.strip())
        self._stream_buf = ""
        if thinking_text and not streamed_thinking:
            # Non-streaming path: reasoning arrived only with the final
            # message, so persist it as its own bubble (live path already
            # shows it in place inside the streaming bubble).
            self._append_thinking(thinking_text)
        if shown and not streamed:
            # Non-streaming path: display the full reply.
            self._append("Assistant", shown)
        elif shown and streamed:
            # Streaming path: text already shown live; add breathing room.
            self._append_html("<div style='height:6px;'></div>")
        if shown:
            code = extract_python_code(shown)
            if code:
                self.last_code = code
                self.apply_btn.setEnabled(True)
                if self.plan_mode:
                    self._append(
                        "System",
                        "Plan ready — review the code, then press “Insert code + Render”.",
                    )
                elif self.preferences["Auto-apply code"] and not tool_calls:
                    self._auto_apply_verdict = self.apply_code(code)
                    if not self._auto_apply_verdict.startswith("OK"):
                        self._append("Tool result", self._auto_apply_verdict[:2000])
                    else:
                        self._request_rendered = True
                else:
                    self._append("System", "Code ready — press “Insert code + Render”.")
        if not tool_calls:
            # No tools requested: if model wrote code as text, offer it.
            # Visual feedback: a text-only code reply that was auto-applied
            # still sends a screenshot so the model can self-correct.
            verdict = getattr(self, "_auto_apply_verdict", "")
            self._auto_apply_verdict = ""
            if verdict and not verdict.startswith("OK"):
                if self.autofix.isChecked() and not self._finalizing:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The script failed validation or rendering. Repair it and return ONE complete "
                                "```python script preserving the design requirements. Error:\n"
                                + verdict
                            ),
                        }
                    )
                    self.iters_left -= 1
                    self._continue_tool_loop()
                else:
                    reason = (
                        "Automatic repair is off."
                        if not self.autofix.isChecked()
                        else "The final attempt failed."
                    )
                    disposition = (
                        "Model displayed for review."
                        if "model displayed for review"
                        in verdict.splitlines()[0].lower()
                        else "Code was not applied."
                    )
                    self._set_busy(
                        False, f"{reason} {disposition} Last error: {verdict[:240]}"
                    )
                return
            if (
                verdict
                and isinstance(verdict, str)
                and verdict.startswith("OK")
                and self._vision_enabled()
                and self.autofix.isChecked()
                and not self._finalizing
            ):
                views = self._capture_design_views()
                if views:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Auto-applied render OK. Screenshot of the current "
                                        "3D view is attached. Inspect it against the request "
                                        "and reply with fixed code if it looks wrong, "
                                        "otherwise confirm it looks correct."
                                    ),
                                },
                                *self._view_blocks(views),
                            ],
                        }
                    )
                    try:
                        thumb = base64.b64encode(views[0][1]).decode("ascii")
                        self._append(
                            "System",
                            "Screenshot sent to the model for visual check.",
                            images=(thumb,),
                        )
                    except Exception:
                        pass
                    self.iters_left -= 1
                    self._update_meter()
                    self._continue_tool_loop()
                    return
            if self._request_rendered:
                message = "Model rendered."
            elif self._finalizing:
                message = "No model was rendered. Review the final response or try a simpler request."
            elif self.last_code:
                message = "Code ready. Use Insert + render to display it."
            else:
                message = "Ready."
            self._set_busy(False, message)
            return
        if self.iters_left <= 0:
            self._append("System", "Max iterations reached.")
            self._set_busy(False)
            return
        results = []
        failures = []
        for block in tool_calls:
            self._turn_steps += 1
            arg = self._tool_arg_preview(block.name, dict(block.input or {}))
            out = self.execute_tool(block.name, dict(block.input or {}))
            failed = self._tool_failed(out)
            if failed:
                failures.append(f"{block.name}: {out[:300]}")
            if (
                block.name == "apply_and_render"
                and out.startswith("OK")
                and not self.plan_mode
            ):
                self._request_rendered = True
                if (
                    self.preferences["Learn reusable CAD skills"]
                    and self.iters_left >= 3
                ):
                    out += (
                        "\nOptional learning: if this completed design contains a useful method "
                        "not covered by existing skills, save it with skill_manage(action='save_draft', ...). "
                        "Then activate the draft only through skill_manage(action='activate', ...), "
                        "which validates its example. Avoid duplicates and finish with a concise design summary."
                    )
            content = out
            # Visual feedback: after a successful render, attach a 3D
            # screenshot so the model sees its own design and can iterate.
            if (
                block.name == "apply_and_render"
                and isinstance(out, str)
                and out.startswith("OK")
                and self._vision_enabled()
            ):
                views = self._capture_design_views()
                if views:
                    verdict = (
                        out + "\nScreenshot of the current 3D view is attached. "
                        "Inspect it against the request and iterate if needed."
                    )
                    content = [{"type": "text", "text": verdict}] + self._view_blocks(
                        views
                    )
                    try:
                        thumb = base64.b64encode(views[0][1]).decode("ascii")
                        self._append(
                            "System",
                            "Screenshot sent to the model for visual check.",
                            images=(thumb,),
                        )
                    except Exception:
                        pass
                else:
                    # Screenshot failed: keep text-only so the loop continues.
                    content = (
                        out
                        + "\n(Visual verification: Not checked; view capture unavailable.)"
                    )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": failed,
                }
            )
            first_line = (out.strip().splitlines() or [""])[0][:160]
            if not failed:
                self._append_step(
                    f"[{self._turn_steps}] {block.name} {arg} → {first_line}", ok=True
                )
            else:
                self._append_step(
                    f"[{self._turn_steps}] {block.name} {arg} → needs fix", ok=False
                )
                self._append("Tool result", out[:2000])
        self._update_meter()
        self.messages.append({"role": "user", "content": results})
        self.iters_left -= 1
        if failures and not self.autofix.isChecked():
            message = "Automatic repair is off. Last tool error: " + failures[-1]
            self._append("System", message)
            self._set_busy(False, message)
            return
        self._continue_tool_loop()

    @pyqtSlot(str)
    def on_llm_err(self, msg):
        if isinstance(self.sender(), LLMWorker) and self.sender() is not self.worker:
            return
        self.worker = None
        self._finish_stream_display()
        self._stream_buf = ""
        self._think_buf = ""
        self._append("Error", msg)
        if "does not support image" in msg:
            try:
                self.vision_check.setChecked(False)
            except Exception:
                pass
            self._append(
                "System",
                "This model can't see images — Vision auto-disabled for this chat. "
                "Switch to a vision-capable model in .env "
                "(e.g. FIREWORKS_MODEL_ID=accounts/fireworks/models/kimi-k3), "
                "restart the app, re-enable Vision, and send the image again.",
            )
        self._set_busy(False, msg)

    # ------------- CAD tools (GUI thread only) -------------
    def _clear_board(self):
        """Remove all rendered objects from the tree + 3D viewer.

        Uses ObjectTree.removeObjects() so sigObjectsRemoved keeps the
        viewer in sync through the normal signal flow. The editor text is
        left untouched — re-render from the editor anytime.
        Returns True when a main window was found.
        """
        try:
            main = self._main()
            if main is None:
                return False
            main.components["object_tree"].removeObjects()
            return True
        except Exception:
            return False

    def _set_editor_and_render(self, code):
        """Write code to the editor and render it, capturing any traceback.

        Returns (ok, message). Shared by apply and session-design restore.
        """
        try:
            main = self._main()
            if main is None:
                return False, "main window not found"
            main.components["editor"].set_text(code)
            dbg = main.components["debugger"]
            captured = {}

            def _cap(exc_info, script):
                captured["exc"] = exc_info

            # Reuse the completed child build; never execute AI code again in Qt.
            dbg.sigTraceback.connect(_cap)
            try:
                cached = self._validated_job
                if (
                    cached is None
                    or cached[0] != code
                    or cached[1] != self.design_requirements
                ):
                    verdict = self._validate_isolated(code)
                    if not verdict.startswith("OK"):
                        return False, verdict
                    cached = self._validated_job
                if self._stop_requested:
                    return False, "Cancelled: no code was applied."
                dbg.display_job(cached[2], code)
            finally:
                dbg.sigTraceback.disconnect(_cap)
            exc = captured.get("exc")
            if exc is None or exc[0] is None:
                return True, "ok"
            import traceback as _tb

            return False, "".join(_tb.format_exception(*exc, limit=5))[-3000:]
        except Exception:
            return False, "Error rendering:\n" + tb.format_exc(limit=5)

    def _board_object_count(self):
        try:
            main = self._main()
            if main is None:
                return 0
            return main.components["object_tree"].CQ.childCount()
        except Exception:
            return 0

    def _reset_editor_to_empty(self):
        """Fresh session: EMPTY editor, nothing rendered, nothing auto-drawn."""
        try:
            main = self._main()
            if main is None:
                return
            editor = main.components["editor"]
            try:
                previous = editor.get_text_with_eol()
            except Exception:
                previous = ""
            if (previous or "").strip() and previous != EMPTY_DESIGN:
                self._push_checkpoint(previous)
            editor.set_text(EMPTY_DESIGN)
        except Exception:
            pass

    def _reset_editor_to_starter(self):
        """Backward-compat alias: now resets to EMPTY (no auto-draw)."""
        return self._reset_editor_to_empty()

    def _restore_session_design(self):
        """Full context switch: board + editor follow the active session.

        Sessions with a saved design get it re-rendered; design-less
        sessions get an EMPTY board and an EMPTY editor (nothing
        auto-drawn, nothing auto-rendered). The replaced editor text is
        pushed as an undo checkpoint first.
        Returns a short status string for the chat.
        """
        code = (self.session_design or "").strip()
        self._stop_requested = False
        self._validated_job = None
        self.verification.invalidate()
        if self.session_id:
            _, meta = self._store.load_session(self.session_id)
            self._set_requirements((meta or {}).get("requirements", []))
        else:
            self._set_requirements([])
        self._clear_board()
        if not code:
            self._reset_editor_to_empty()
            return "Empty design — editor and board cleared, nothing rendered yet."
        try:
            main = self._main()
            if main is None:
                return "Main window not found — board cleared."
            editor = main.components["editor"]
            try:
                previous = editor.get_text_with_eol()
            except Exception:
                previous = ""
            if previous and previous != code:
                self._push_checkpoint(previous)
        except Exception:
            pass
        ok, msg = self._set_editor_and_render(code)
        if ok:
            n = self._board_object_count()
            return f"Design restored — {n} object(s) on the board."
        self._append("Error", "Saved design failed to render:\n" + msg[:2000])
        return "Saved design failed to render (see traceback). Board cleared."

    def _probe_workdir(self):
        """Script dir for the isolated probe (mirrors Debugger path setup)."""
        try:
            main = self._main()
            if main is None:
                return None
            filename = main.components["editor"].filename
            if filename:
                parent = Path(filename).absolute().parent
                if parent.is_dir():
                    return str(parent)
        except Exception:
            pass
        return None

    def _validate_isolated(self, code):
        """Cancellable child build; successful geometry is cached for display."""
        if self._stop_requested:
            return "Error: Cancelled: CAD job stopped."
        if (
            self._validated_job is not None
            and self._validated_job[0] == code
            and self._validated_job[1] == self.design_requirements
        ):
            result = self._validated_job[2]
            self._show_verification(result["report"], code=code)
            return (
                f"OK: reused the validated build, {len(result['objects'])} object(s).\n"
                + report_text(result["report"])
            )
        try:
            timeout = int(self.preferences["Pre-render timeout (s)"])
        except Exception:
            timeout = ISOLATED_PROBE_TIMEOUT
        try:
            self.status.setText("Validating geometry in an isolated process…")
        except Exception:
            pass
        result = self._run_cad(
            dict(
                code=code,
                requirements=self.design_requirements,
                workdir=self._probe_workdir(),
            ),
            timeout=timeout,
        )
        if not result.get("ok"):
            self._validated_job = None
            error = result.get("error", "CAD build failed")
            if result.get("traceback"):
                from ..ai.cad_agent import validation_failure

                return validation_failure(
                    code,
                    result["traceback"],
                    result.get("line"),
                    error,
                    result.get("version", ""),
                )
            return (
                error
                if error.startswith(("CRASH:", "TIMEOUT:", "Error:"))
                else "Error: " + error
            )
        self._validated_job = (code, list(self.design_requirements), result)
        self._show_verification(result["report"], code=code)
        return (
            f"OK: isolated build created {len(result['objects'])} object(s).\n"
            + report_text(result["report"])
        )

    def _run_cad(self, request, timeout=180, shapes=None):
        from .cad_job import run_cad_job

        if self._operation_active:
            return {"ok": False, "error": "Error: another CAD operation is running."}
        self._operation_active = True
        self._set_busy(True, "Building and verifying geometry")
        try:
            return run_cad_job(
                request,
                owner=self,
                timeout=timeout,
                cancelled=lambda: self._stop_requested,
                shapes=shapes,
            )
        except Exception as error:
            return {
                "ok": False,
                "error": f"Error: CAD operation could not run: {type(error).__name__}: {error}",
            }
        finally:
            self._operation_active = False
            # The parent AI turn keeps its controls locked between tools.
            if self.iters_left <= 0 or self._stop_requested:
                self._set_busy(
                    False,
                    "Stopped." if self._stop_requested else "CAD operation finished.",
                )

    def _set_requirements(self, requirements):
        self.design_requirements = normalize_requirements(requirements)
        self._validated_job = None
        self.verification_report = None
        self._verification_code = None
        self._visual_capture_revision = None
        self.verification.set_requirements(self.design_requirements)

    def _show_verification(self, report, code=None):
        self.verification_report = report
        self._verification_code = (
            self._editor_text() if code is None else code
        ).replace("\r\n", "\n")
        self.verification.set_report(report)

    def _on_editor_changed(self):
        current_code = self._editor_text().replace("\r\n", "\n")
        if (
            self._verification_code is not None
            and self._verification_code == current_code
        ):
            return  # syntax highlighting and theme updates can emit textChanged
        self.verification_report = None
        self._verification_code = None
        self._visual_capture_revision = None
        self.verification.invalidate()
        if (
            self._validated_job is not None
            and self._validated_job[0].replace("\r\n", "\n") != current_code
        ):
            self._validated_job = None

    def verify_design(self):
        code = self._editor_text()
        if not code.strip():
            return "Error: no design code to verify."
        verdict = self._validate_isolated(code)
        if (
            verdict.startswith("OK")
            and self.verification_report
            and not self.verification_report["ok"]
        ):
            return "Error: design verification failed.\n" + report_text(
                self.verification_report
            )
        return verdict

    @pyqtSlot()
    def on_verify_design(self):
        if self.worker is not None or self._operation_active:
            return
        self._stop_requested = False
        output = self.verify_design()
        self._append("Verification", output)
        self._set_busy(
            False, "Verification finished. Review Passed / Failed / Not checked."
        )

    def execute_tool(self, name, args):
        try:
            if self._stop_requested:
                return "Error: Cancelled: no further tools were executed."
            from ..ai.recovery import cad_argument_error

            argument_error = cad_argument_error(name, args)
            if argument_error:
                return "Error: " + argument_error
            if name == "get_editor_code":
                return self.get_editor_code()
            if name == "set_design_requirements":
                requirements = normalize_requirements(args.get("requirements", []))
                if (
                    self._requirements_recorded_this_turn
                    and requirements != self.design_requirements
                ):
                    return "Error: requirements are fixed for this request. Repair the geometry without weakening the checks. The user can edit Checks or send updated requirements."
                self._set_requirements(requirements)
                self._requirements_recorded_this_turn = True
                return (
                    "OK: requirements recorded for independent geometry measurements.\n"
                    + str(self.design_requirements)
                )
            if name == "verify_design":
                return self.verify_design()
            if name == "record_visual_review":
                if (
                    self._visual_capture_revision != self._editor_text()
                    or self.verification_report is None
                ):
                    return "Error: capture current design views before recording a visual review."
                status, notes = args.get("status"), args.get("notes", "")
                if (
                    status not in ("Passed", "Failed", "Not checked")
                    or not isinstance(notes, str)
                    or not notes.strip()
                ):
                    return "Error: provide a review status and notes about the inspected views."
                from ..ai.verification import row

                self.verification_report["rows"] = [
                    r
                    for r in self.verification_report["rows"]
                    if r["label"] != "AI visual review"
                ]
                self.verification_report["rows"].append(
                    row(
                        "AI visual review",
                        status,
                        detail=notes[:4000]
                        + " This is the AI's visual assessment, not a measured geometry check.",
                    )
                )
                self.verification_report["ok"] = not any(
                    r["status"] == "Failed" for r in self.verification_report["rows"]
                )
                self.verification.set_report(self.verification_report)
                return (
                    "Error: visual review found discrepancies. "
                    if status == "Failed"
                    else "OK: visual review recorded. "
                ) + notes[:4000]
            if name == "validate_cad_code":
                # Crash-safe: runs in a child process (a native OCCT
                # segfault on complex geometry must not kill the app).
                return self._validate_isolated(args.get("code", ""))
            if name == "apply_and_render":
                code = args.get("code", "")
                if self.plan_mode:
                    # Plan mode (Codex-style): propose, don't touch the editor.
                    self.last_code = code
                    self.apply_btn.setEnabled(True)
                    verdict = self._validate_isolated(code)
                    return verdict + "\n(plan mode: validated only, not applied)"
                self.last_code = code
                self.apply_btn.setEnabled(True)
                return self.apply_code(code)
            if name == "get_examples":
                return self.get_examples(args.get("name", ""))
            if name == "skill_view":
                return self._skills.view(
                    args.get("name", ""),
                    args.get("file"),
                    draft=bool(args.get("draft", False)),
                )
            if name == "skill_manage":
                if self.plan_mode or not self.preferences["Learn reusable CAD skills"]:
                    return "Error: skill learning is disabled in Plan mode or Preferences. Existing skills can still be read."
                if self._skills.validator is None:
                    self._skills.validator = self._validate_skill
                return self._skills.manage(
                    args.get("action", ""),
                    args.get("name", ""),
                    args.get("description", ""),
                    args.get("instructions", ""),
                    args.get("example_code", ""),
                )
            if name == "export_model":
                return self.export_model(
                    args.get("format", "stl"), args.get("filename", "")
                )
            if name == "spawn_subagent":
                return self._spawn_subagent_tool(args)
            return f"Error: unknown tool {name}"
        except Exception:
            return "Error:\n" + tb.format_exc(limit=5)

    def _validate_skill(self, code):
        result = self._run_cad(
            dict(
                code=code,
                requirements=[],
                assembly=False,
                workdir=self._probe_workdir(),
            ),
            timeout=int(self.preferences["Pre-render timeout (s)"]),
        )
        return (
            "OK: skill example produced valid geometry."
            if result.get("ok")
            else "Error: " + result.get("error", "Skill validation failed")
        )

    def _capture_screenshot(self):
        """Capture the current 3D viewer image as JPEG bytes.

        Falls back to a Qt widget grab if the OCC view is not available.
        Returns bytes or an error string.
        """
        try:
            main = self._main()
            if main is None:
                return "Error: main window not found."
            viewer = main.components.get("viewer")
            if viewer is None:
                return "Error: 3D viewer not found."
            view = viewer._get_view()
            if view is None:
                return "Error: 3D view is not initialized."
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                view.Dump(tmp_path)
                img = QImage(tmp_path)
                if img.isNull():
                    return f"Error: could not read screenshot from {tmp_path}"
                if max(img.width(), img.height()) > 1024:
                    img = img.scaled(
                        1024, 1024, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                from PyQt5.QtCore import QBuffer, QIODevice

                buf = QBuffer()
                buf.open(QIODevice.WriteOnly)
                if not img.save(buf, "JPEG", 80):
                    return "Error: could not encode screenshot to JPEG."
                return bytes(buf.data())
            finally:
                try:
                    import os

                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception:
            return "Error capturing screenshot:\n" + tb.format_exc(limit=5)

    def _vision_enabled(self):
        """Whether automatic screenshot feedback is on (Vision toggle)."""
        try:
            return bool(self.vision_check.isChecked())
        except Exception:
            return True

    def _capture_design_views(self):
        """Capture labelled views and restore the user's exact camera."""
        from OCP.Graphic3d import Graphic3d_Camera
        from OCP.V3d import V3d_XposYnegZpos, V3d_Yneg, V3d_Zpos, V3d_Xpos

        views = []
        view = None
        camera = None
        try:
            main = self._main()
            if main is None:
                return []
            viewer = main.components["viewer"]
            view = viewer._get_view()
            if view is None:
                return []
            camera = Graphic3d_Camera(view.Camera())
            for label, orientation in [
                ("Isometric", V3d_XposYnegZpos),
                ("Front", V3d_Yneg),
                ("Top", V3d_Zpos),
                ("Side", V3d_Xpos),
            ]:
                if self._stop_requested:
                    break
                view.SetProj(orientation)
                # The app's fit excludes axis helpers, which otherwise make
                # small parts occupy only a tiny fraction of the image.
                if hasattr(viewer, "fit"):
                    viewer.fit()
                else:
                    view.FitAll()
                view.Redraw()
                capture = self._capture_screenshot()
                if isinstance(capture, bytes):
                    views.append((label, capture))
        except Exception:
            pass
        finally:
            if view is not None and camera is not None:
                view.Camera().Copy(camera)
                view.Redraw()
        if not views:
            capture = self._capture_screenshot()
            if isinstance(capture, bytes):
                views.append(("Current view", capture))
        if self.verification_report is not None:
            from ..ai.verification import row

            self._visual_capture_revision = self._editor_text() if views else None
            self.verification_report["rows"] = [
                r
                for r in self.verification_report["rows"]
                if r["label"] != "AI visual review"
            ]
            self.verification_report["rows"].append(
                row(
                    "AI visual review",
                    "Not checked",
                    detail=(
                        ", ".join(label for label, _ in views)
                        + " captured for AI review; capture alone does not certify visual correctness."
                        if views
                        else "No screenshots could be captured."
                    ),
                )
            )
            self.verification.set_report(self.verification_report)
        return views

    def _view_blocks(self, views):
        blocks = []
        for label, jpeg in views:
            blocks.append({"type": "text", "text": label + " view"})
            blocks.append(self._screenshot_image_block(jpeg))
        return blocks

    @staticmethod
    def _screenshot_image_block(jpeg_bytes):
        """Anthropic image block for raw JPEG bytes (vision feedback)."""
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(jpeg_bytes).decode("ascii"),
            },
        }

    def _tool_result_with_screenshot(self, text, jpeg_bytes):
        """Build a tool_result content list: text verdict + 3D screenshot.

        Keeps the Anthropic tool_use/tool_result pairing intact while letting
        the model SEE what it just rendered. Falls back to plain text when
        no screenshot bytes are available.
        """
        if not jpeg_bytes:
            return text
        return [
            {"type": "text", "text": text},
            self._screenshot_image_block(jpeg_bytes),
        ]

    def _spawn_subagent_tool(self, args):
        """Run a focused subagent off the GUI thread and return its text."""
        purpose = args.get("purpose", "explain")
        task = args.get("task", "")
        if not task:
            return "Error: subagent task is empty."

        from PyQt5.QtCore import QThread, pyqtSignal

        class SubagentWorker(QThread):
            finished_ok = pyqtSignal(object)
            finished_err = pyqtSignal(str)

            def __init__(
                self,
                parent,
                purpose,
                task,
                messages,
                model,
                effort,
                client,
                image_bytes=None,
                current_code="",
                skills=None,
            ):
                super().__init__(parent)
                self.purpose = purpose
                self.task = task
                self.messages = messages
                self.model = model
                self.effort = effort
                self.client = client
                self.image_bytes = image_bytes
                self.current_code = current_code
                self.skills = skills

            def run(self):
                try:
                    if self.purpose == "code":
                        result = run_code_subagent(
                            request=self.task,
                            current_code=self.current_code,
                            parent_messages=self.messages,
                            model=self.model,
                            effort=self.effort,
                            client=self.client,
                            skills=self.skills,
                            cancelled=self.isInterruptionRequested,
                        )
                    else:
                        result = run_subagent(
                            task=self.task,
                            parent_messages=self.messages,
                            model=self.model,
                            effort=self.effort,
                            image_bytes=self.image_bytes,
                            client=self.client,
                            skills=self.skills,
                            cancelled=self.isInterruptionRequested,
                        )
                    self.finished_ok.emit(result)
                except Exception as e:
                    self.finished_err.emit(f"{type(e).__name__}: {e}")

        image_bytes = None
        if purpose == "visual_validate":
            capture = self._capture_screenshot()
            if isinstance(capture, str) and capture.startswith("Error:"):
                return capture
            image_bytes = capture
            if not task.lower().startswith("analyze"):
                task = (
                    "Analyze the attached screenshot of the current 3D view "
                    "and compare it to the design request. "
                    "Report what looks correct and what needs modification. "
                    "If changes are needed, include a corrected ```python "
                    "fenced block with the full build123d script.\n\n" + task
                )

        self._set_busy(True, "Running subagent")
        worker = SubagentWorker(
            self,
            purpose,
            task,
            list(self.messages),
            getattr(self, "_model", None),
            self.effort,
            getattr(self, "_client", None),
            image_bytes=image_bytes,
            current_code=self._editor_text(),
            skills=self._skills,
        )
        result_holder = {"value": None}
        from PyQt5.QtCore import QEventLoop

        loop = QEventLoop()

        def on_ok(res):
            result_holder["value"] = res
            loop.quit()

        def on_err(msg):
            result_holder["value"] = {
                "ok": False,
                "text": msg,
                "usage": {"in": 0, "out": 0},
            }
            loop.quit()

        worker.finished_ok.connect(on_ok)
        worker.finished_err.connect(on_err)
        self._retain_worker(worker)
        self._operation_active = True
        worker.start()
        timer = QTimer()
        timer.setInterval(50)

        def cancel():
            if self._stop_requested:
                worker.requestInterruption()
                loop.quit()

        timer.timeout.connect(cancel)
        timer.start()
        deadline = QTimer()
        deadline.setSingleShot(True)

        def timed_out():
            worker.requestInterruption()
            result_holder["value"] = {
                "ok": False,
                "text": "Subagent exceeded its 240s time budget.",
                "usage": {},
            }
            loop.quit()

        deadline.timeout.connect(timed_out)
        deadline.start(240000)
        loop.exec_()
        timer.stop()
        deadline.stop()
        self._operation_active = False
        if self._stop_requested:
            return "Error: subagent cancelled."

        result = result_holder["value"] or {
            "ok": False,
            "text": "No result",
            "usage": {"in": 0, "out": 0},
        }
        usage = result.get("usage") or {"in": 0, "out": 0}
        self.sess_usage["in"] += int(usage.get("in", 0))
        self.sess_usage["out"] += int(usage.get("out", 0))
        self._update_meter()

        if not result.get("ok"):
            return f"Subagent failed: {result.get('text', '')}"

        text = result.get("text", "")
        if purpose == "code":
            code = extract_python_code(text)
            if code:
                self.last_code = code
                self.apply_btn.setEnabled(True)
        elif purpose == "visual_validate":
            code = extract_python_code(text)
            if code:
                self.last_code = code
                self.apply_btn.setEnabled(True)
        return text or "(subagent returned no text)"

    def get_editor_code(self):
        try:
            main = self._main()
            if main is None:
                return "Error: main window not found."
            code = main.components["editor"].get_text_with_eol()
            return code[:20000] if code else "(editor empty)"
        except Exception as e:
            return f"Error reading editor: {e}"

    def get_examples(self, name=""):
        try:
            from ..ai.cad_agent import read_example

            main = self._main()
            ex_dir = main._examples_dir() if main is not None else None
            return read_example(name, ex_dir)
        except Exception as e:
            return f"Error listing examples: {e}"

    def apply_code(self, code):
        """Set editor text + trigger debugger.render(), capture traceback.

        Pushes the previous editor text as an undo checkpoint first.
        On success the code becomes this session's design (restored on switch).

        Validation must succeed before the editor or viewer is changed.
        Failures are returned to the generation loop so it can repair the code.
        """
        if not isinstance(code, str) or not code.strip():
            return "Error: empty code"
        probe = self._validate_isolated(code)
        if not probe.startswith("OK"):
            self._append("Error", probe)
            try:
                self.status.setText("Validation failed; code was not applied.")
            except Exception:
                pass
            return probe
        try:
            main = self._main()
            if main is None:
                return "Error: main window not found, cannot apply code."
            editor = main.components["editor"]
            try:
                previous = editor.get_text_with_eol()
            except Exception:
                previous = ""
            if previous and previous != code:
                self._push_checkpoint(previous)
            stat = self._diffstat(previous or "", code)
            ok, msg = self._set_editor_and_render(code)
            if ok:
                suffix = f" (editor {stat})" if stat else ""
                self.session_design = code
                self._save_session()
                self._append_success_card(code)
                report = self.verification_report
                if report and not report.get("ok"):
                    return (
                        "Error: design verification failed; model displayed for review. Repair failed checks without changing their expected values.\n"
                        + report_text(report)
                    )
                return (
                    "OK: code applied and rendered, no traceback."
                    + suffix
                    + ("\n" + report_text(report) if report else "")
                )
            return "Rendered WITH traceback:\n" + msg
        except Exception:
            return "Error applying code:\n" + tb.format_exc(limit=5)

    def _append_success_card(self, code):
        """Show a green status card when the model renders successfully."""
        colors = theme_for(self._app_is_dark())
        bg, fg, border = colors.selection, colors.accent, colors.border
        lines = len(code.splitlines())
        chars = len(code)
        stamp = self._now()
        title = (
            "Model rendered; checks need review."
            if self.verification_report and not self.verification_report.get("ok")
            else "Model created successfully!"
        )
        card = (
            f"<div style='margin:8px 2px 12px;'>"
            f"<table width='100%' cellspacing='0' cellpadding='0'><tr><td align='center'>"
            f"<table width='92%' cellspacing='0' cellpadding='12' "
            f"style='background:{bg};border:1px solid {border};border-radius:12px;'>"
            f"<tr><td>"
            f"<div style='font-size:14px;font-weight:800;color:{fg};margin-bottom:4px;'>"
            f"{title}</div>"
            f"<div style='font-size:12px;color:{fg};line-height:1.45;'>"
            f"Your build123d model was generated and rendered in the 3D view. "
            f"({lines} lines, {chars} chars) · {stamp}"
            f"</div></td></tr></table></td></tr></table></div>"
        )
        self._append_html(card)

    @pyqtSlot()
    def on_apply_last(self):
        if self.last_code and not self._operation_active and self.worker is None:
            self._stop_requested = False
            out = self.apply_code(self.last_code)
            self._append("System", out)
            self._set_busy(False, out.splitlines()[0])

    # ------------- export -------------
    def _collect_shapes(self):
        """All shapes currently in the object tree (for export)."""
        main = self._main()
        if main is None:
            return []
        tree = main.components["object_tree"]
        shapes = []

        def _walk(item):
            if getattr(item, "shape", None) is not None:
                shapes.append(item.shape)
            for i in range(item.childCount()):
                _walk(item.child(i))

        cq_root = getattr(tree, "CQ", None)
        if cq_root is not None:
            _walk(cq_root)
        return shapes

    def export_model(self, fmt="stl", filename="", shapes=None, precision=None):
        """Agent-callable export. Defaults to exports/model.<ext> in cwd."""
        fmt = (fmt or "stl").lower().lstrip(".")
        if fmt not in ("stl", "step", "brep"):
            return f"Error: unsupported format '{fmt}'. Use stl/step/brep."
        try:
            shapes = self._collect_shapes() if shapes is None else shapes
            if not shapes:
                return "Error: no models to export. Render code first."
            if not filename:
                out_dir = Path.cwd() / "exports"
                out_dir.mkdir(parents=True, exist_ok=True)
                filename = str(out_dir / f"model.{fmt}")
            result = self._run_cad(
                dict(
                    requirements=[],
                    assembly=False,
                    export={
                        "format": fmt,
                        "filename": str(Path(filename).resolve()),
                        "precision": 0.1 if precision is None else float(precision),
                    },
                ),
                timeout=int(self.preferences["Pre-render timeout (s)"]),
                shapes={f"object{i + 1}": shape for i, shape in enumerate(shapes)},
            )
            if not result.get("ok"):
                return "Error exporting: " + result.get("error", "CAD export failed")
            report = result["report"]
            if self.verification_report:
                exported_rows = [
                    r
                    for r in report["rows"]
                    if "export" in r["label"].lower()
                    or "reopened" in r["label"].lower()
                    or r["label"].startswith("STL ")
                ]
                report = dict(
                    self.verification_report,
                    rows=self.verification_report["rows"] + exported_rows,
                )
                report["ok"] = not any(r["status"] == "Failed" for r in report["rows"])
            self._show_verification(report)
            prefix = (
                "OK:"
                if result["report"]["ok"]
                else "Error: export verification failed."
            )
            return (
                f"{prefix} exported {len(shapes)} object(s) to {filename}\n"
                + report_text(result["report"])
            )
        except Exception:
            return "Error exporting:\n" + tb.format_exc(limit=5)

    @pyqtSlot()
    def on_export_button(self, fmt):
        if self._operation_active or self.worker is not None:
            return
        shapes = self._collect_shapes()
        if not shapes:
            self._append("System", "Nothing to export — render a model first.")
            return
        fname = get_save_filename(fmt)
        if not fname:
            return
        try:
            self._stop_requested = False
            output = self.export_model(fmt, fname)
            self._append("System", output)
            self._set_busy(False, output.splitlines()[0])
        except Exception as e:
            self._append("Error", f"Export failed: {e}")

    # ------------- persistence -------------
    # NOTE: model is intentionally NOT persisted - it always comes from .env,
    # so a stale picker value can never override it again (caused 404s).
    # Old "AI Chat/model" keys left in QSettings are ignored.
    def saveComponentState(self, store):
        store.setValue(self.name + "/stream", self.stream_check.isChecked())
        store.setValue(self.name + "/plan", self.plan_check.isChecked())
        store.setValue(self.name + "/vision", self.vision_check.isChecked())
        store.setValue(self.name + "/effort", self.effort)
        self._save_session()
        if self.session_id is not None:
            store.setValue(self.name + "/session", self.session_id)
        else:
            try:
                store.remove(self.name + "/session")
            except Exception:
                pass

    def restoreComponentState(self, store):
        stream = store.value(self.name + "/stream")
        if stream is not None:
            self.stream_check.setChecked(str(stream).lower() in ("true", "1"))
        vision = store.value(self.name + "/vision")
        if vision is not None:
            self.vision_check.setChecked(str(vision).lower() in ("true", "1"))
        plan = store.value(self.name + "/plan")
        if plan is not None and str(plan).lower() in ("true", "1"):
            self._set_plan(True)
        effort = store.value(self.name + "/effort")
        if effort is not None:
            self._set_effort(effort)
        else:
            self._refresh_model_lbl()
        # Restore the last active chat session (persisted transcripts).
        active = store.value(self.name + "/session")
        entries = self._store.list_sessions()
        target = None
        if active and any(e.get("id") == active for e in entries):
            target = active
        elif entries:
            target = entries[0]["id"]
        if target is not None:
            messages, meta = self._store.load_session(target)
            if messages is not None:
                self.session_id = target
                self.session_title = (meta or {}).get("title") or DEFAULT_TITLE
                self.messages = messages
                usage = (meta or {}).get("usage") or {}
                self.sess_usage = {
                    "in": int(usage.get("in", 0) or 0),
                    "out": int(usage.get("out", 0) or 0),
                }
                self.session_design = (meta or {}).get("design_code", "") or ""
        if self.session_id is None:
            # Fresh start with no history: lazy unsaved chat, no junk file.
            self.session_id = None
            self.session_title = DEFAULT_TITLE
        self._refresh_session_combo()
        self._render_history()
        self._update_meter()
        # Board + editor always follow the active session: a saved design is
        # re-rendered, otherwise we clear to EMPTY (overrides fill_dummy so
        # a new chat never shows an auto-drawn box on startup).
        note = self._restore_session_design()
        if (self.session_design or "").strip():
            self._append("System", note)
