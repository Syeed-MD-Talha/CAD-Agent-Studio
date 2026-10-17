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

from PyQt5.QtCore import Qt, QThread, QTimer, QStringListModel, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QImage, QPixmap
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
)
from ..ai.subagent import run_subagent, run_code_subagent
from ..ai.sessions import SessionStore, autotitle, DEFAULT_TITLE, is_auto_title


# Legacy starter snippet (kept for backward compatibility / tests).
# NOTE: new sessions no longer insert this — every fresh chat starts with
# an EMPTY editor and an EMPTY 3D board (nothing auto-drawn, nothing
# auto-rendered). The user (or the agent via apply_and_render) explicitly
# creates the first design.
STARTER_CODE = (
    "from build123d import *\n"
    "result = Box(3, 3, 0.5)\n"
    "show_object(result)"
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
    "/compact": "Show what auto-compaction sends vs stores",
    "/model": "Show the active model (from .env)",
    "/effort": "Show or set reasoning effort: /effort [low|medium|high|max]",
    "/help": "List all slash commands",
}


class LLMWorker(QThread):
    """One LLM turn in a background thread (network only, no Qt access)."""

    finished_ok = pyqtSignal(object)  # Anthropic response
    finished_err = pyqtSignal(str)

    def __init__(self, client, model, messages, parent=None, thinking=None,
                 max_tokens=8000):
        super().__init__(parent)
        self.client = client
        self.model = model
        # Shallow copy; blocks are read-only during the call.
        self.messages = list(messages)
        self.thinking = thinking
        self.max_tokens = max_tokens

    def run(self):
        try:
            kwargs = {}
            if self.thinking is not None:
                kwargs["thinking"] = self.thinking
            resp = self.client.messages.create(
                model=self.model,
                system=__import__(
                    "cq_editor.ai.cad_agent", fromlist=["SYSTEM_PROMPT"]
                ).SYSTEM_PROMPT,
                messages=self.messages,
                tools=__import__(
                    "cq_editor.ai.cad_agent", fromlist=["TOOLS"]
                ).TOOLS,
                max_tokens=self.max_tokens,
                **kwargs,
            )
            self.finished_ok.emit(resp)
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")


class StreamWorker(QThread):
    """Streaming LLM turn: emits text AND thinking deltas live, then final.

    Uses client.messages.stream() so the chat shows tokens as they arrive.
    Iterates parsed stream events (not just text_stream) so extended-thinking
    models (e.g. glm with a thinking budget) show their reasoning live too.
    Falls back to non-streaming create() when the gateway does not support
    streaming (some OpenAI-compatible endpoints).
    """

    text_delta = pyqtSignal(str)
    thinking_delta = pyqtSignal(str)
    finished_ok = pyqtSignal(object)  # final message (has .content blocks)
    finished_err = pyqtSignal(str)

    def __init__(self, client, model, messages, parent=None, thinking=None,
                 max_tokens=8000):
        super().__init__(parent)
        self.client = client
        self.model = model
        self.messages = list(messages)
        self.thinking = thinking
        self.max_tokens = max_tokens

    def run(self):
        try:
            from cq_editor.ai.cad_agent import SYSTEM_PROMPT, TOOLS

            kwargs = {}
            if self.thinking is not None:
                kwargs["thinking"] = self.thinking
            try:
                with self.client.messages.stream(
                    model=self.model,
                    system=SYSTEM_PROMPT,
                    messages=self.messages,
                    tools=TOOLS,
                    max_tokens=self.max_tokens,
                    **kwargs,
                ) as stream:
                    for event in stream:
                        etype = getattr(event, "type", "")
                        if etype == "text":
                            self.text_delta.emit(event.text)
                        elif etype == "thinking":
                            self.thinking_delta.emit(event.thinking)
                    final = stream.get_final_message()
                self.finished_ok.emit(final)
            except Exception as stream_err:
                # Fallback for gateways without .stream support.
                if "stream" in type(stream_err).__name__.lower() or "stream" in str(
                    stream_err
                ).lower() or "no attribute" in str(stream_err).lower():
                    resp = self.client.messages.create(
                        model=self.model,
                        system=SYSTEM_PROMPT,
                        messages=self.messages,
                        tools=TOOLS,
                        max_tokens=self.max_tokens,
                        **kwargs,
                    )
                    self.finished_ok.emit(resp)
                else:
                    raise
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")


class ChatWidget(QWidget, ComponentMixin):
    name = "AI Chat"

    preferences = Parameter.create(
        name="Preferences",
        children=[
            {"name": "Max tool iterations", "type": "int", "value": 6, "limits": (1, 20)},
            {"name": "Auto-apply code", "type": "bool", "value": True},
            {"name": "Isolated pre-render", "type": "bool", "value": True},
            {"name": "Pre-render timeout (s)", "type": "int", "value": 180, "limits": (10, 1800)},
        ],
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        ComponentMixin.__init__(self)

        self.messages = []  # Anthropic conversation history
        self.last_code = ""
        self.worker = None
        self.iters_left = 0
        self._pending_traceback = None
        self._stream_buf = ""
        self._think_buf = ""
        self._stream_active = False
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
        hlay = QHBoxLayout(header)
        hlay.setContentsMargins(12, 10, 8, 10)
        hlay.setSpacing(10)
        self.title_icon = QLabel(header)
        self.title_icon.setObjectName("brandIcon")
        if _HAS_QTA:
            try:
                self.title_icon.setPixmap(
                    qta.icon("fa5s.cube", color="#2563eb").pixmap(26, 26)
                )
            except Exception:
                self.title_icon.setText("◈")
        else:
            self.title_icon.setText("◈")
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel("CAD Assistant", header)
        self.title_label.setObjectName("chatTitle")
        self.subtitle_label = QLabel("AI part designer · build123d", header)
        self.subtitle_label.setObjectName("chatSubtitle")
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.subtitle_label)
        self.provider_pill = QLabel("● fireworks", header)
        self.provider_pill.setObjectName("providerPill")
        self.provider_pill.setToolTip("Active provider (from .env)")
        self.mode_pill = QLabel("PLAN", header)
        self.mode_pill.setObjectName("modePill")
        self.mode_pill.setToolTip("Plan mode: code is validated, never auto-applied (/plan to toggle)")
        self.mode_pill.setVisible(False)
        hlay.addWidget(self.title_icon)
        hlay.addLayout(title_box, 1)
        hlay.addWidget(self.mode_pill)
        hlay.addWidget(self.provider_pill)
        self.clear_btn = QPushButton(header)
        self.clear_btn.setObjectName("iconBtn")
        self.clear_btn.setToolTip("Start new chat")
        self.clear_btn.setFixedSize(30, 30)
        if _HAS_QTA:
            try:
                self.clear_btn.setIcon(qta.icon("fa5s.broom"))
            except Exception:
                self.clear_btn.setText("✕")
        else:
            self.clear_btn.setText("✕")
        hlay.addWidget(self.clear_btn)
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
        self.preview_hint = QLabel("Reference images — click one to remove:", self.preview_bar)
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
        self.attach_btn.setToolTip("Attach reference image (PNG/JPG — max 3 per message)")
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
            "Describe the 3D part you need…  (Enter to send · Shift+Enter for new line)"
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
        self.apply_btn = QPushButton("Insert code + Render", self)
        self.apply_btn.setObjectName("applyBtn")
        if _HAS_QTA:
            try:
                self.apply_btn.setIcon(qta.icon("fa5s.cube"))
            except Exception:
                pass
        self.apply_btn.setEnabled(False)
        self.apply_btn.setToolTip("Write the last generated code into the editor and render it.")
        self.undo_btn = QPushButton("Undo", self)
        self.undo_btn.setObjectName("expBtn")
        if _HAS_QTA:
            try:
                self.undo_btn.setIcon(qta.icon("fa5s.undo"))
            except Exception:
                pass
        self.undo_btn.setEnabled(False)
        self.undo_btn.setToolTip("Restore editor code from before the last apply (/undo).")
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
        self.autofix = QCheckBox("Self-fix", self)
        self.autofix.setChecked(True)
        self.autofix.setToolTip("Agentic loop: validate + feed tracebacks back to the model.")
        self.plan_check = QCheckBox("Plan", self)
        self.plan_check.setChecked(False)
        self.plan_check.setToolTip("Plan mode: validate code, never auto-apply (/plan).")
        self.stream_check = QCheckBox("Stream", self)
        self.stream_check.setChecked(True)
        self.stream_check.setToolTip("Show tokens as they arrive (falls back if unsupported).")
        self.vision_check = QCheckBox("Vision", self)
        self.vision_check.setChecked(True)
        self.vision_check.setToolTip(
            "Visual feedback: after each successful render, send a 3D screenshot "
            "to the model so it can verify and refine the design."
        )
        self.effort = DEFAULT_EFFORT
        self.effort_combo = QComboBox(self)
        self.effort_combo.addItems(["Low", "Medium", "High", "Max"])
        self.effort_combo.setCurrentText("Medium")
        self.effort_combo.setToolTip(self._effort_tooltip())
        self.model_lbl = QLabel("Model: from .env", self)
        self.model_lbl.setObjectName("muted")
        self.model_lbl.setToolTip(
            "Model id is read from MODEL_ID / FIREWORKS_MODEL_ID in .env. "
            "Restart the app after changing .env."
        )
        toggles_row = QHBoxLayout()
        toggles_row.setSpacing(12)
        toggles_row.addWidget(self.model_lbl, 1)
        toggles_row.addWidget(self.effort_combo)
        toggles_row.addWidget(self.autofix)
        toggles_row.addWidget(self.plan_check)
        toggles_row.addWidget(self.stream_check)
        toggles_row.addWidget(self.vision_check)
        chat_area.addLayout(toggles_row)

        # status
        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        self.status_dot = QLabel("●", self)
        self.status_dot.setObjectName("statusDot")
        self.status = QLabel("Add FIREWORKS_API_KEY to .env to start.", self)
        self.status.setObjectName("statusLbl")
        self.status.setWordWrap(True)
        self.meter_lbl = QLabel("", self)
        self.meter_lbl.setObjectName("muted")
        self.meter_lbl.setToolTip("Context estimate · last-turn tokens · session totals (/usage)")
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
        # Slash-command completion (Claude/Codex "/" popup pattern).
        self.slash_completer = QCompleter(self)
        self.slash_completer.setModel(
            QStringListModel([f"{cmd}  —  {desc}" for cmd, desc in SLASH_COMMANDS.items()])
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
            self._apply_styles()
            self._render_history()
        except Exception:
            pass

    def _apply_styles(self):
        dark = self._app_is_dark()
        if dark:
            hist_bg, card, ink, muted = "#101728", "#1a2342", "#e8edf7", "#93a0b8"
            user_bg, asst_border = "#2f6fed", "#33436b"
            subtle = "#1a2342"
        else:
            hist_bg, card, ink, muted = "#eef1f6", "#ffffff", "#0f1f38", "#64748b"
            user_bg, asst_border = "#2563eb", "#dfe5ee"
            subtle = "#e7ecf3"
        self.setStyleSheet(
            f"""
            #chatHeader {{ background: {card}; border: 1px solid {asst_border}; border-radius: 16px; }}
            #brandIcon {{ font-size: 22px; color: #2563eb; }}
            #chatTitle {{ font-size: 15px; font-weight: 800; color: {ink}; letter-spacing: 0.2px; }}
            #chatSubtitle {{ font-size: 11px; color: {muted}; }}
            #providerPill {{ background: #16a34a18; color: #16a34a; border: 1px solid #16a34a44;
                             border-radius: 10px; padding: 3px 10px; font-size: 11px; font-weight: 700; }}
            #modePill {{ background: #7c3aed18; color: #7c3aed; border: 1px solid #7c3aed55;
                        border-radius: 10px; padding: 3px 10px; font-size: 11px; font-weight: 800; }}
            #chatHistory {{ background: {hist_bg}; border: 1px solid {asst_border}; border-radius: 16px; padding: 10px; font-size: 13px; }}
            #chatHistory QScrollBar:vertical {{ width: 8px; background: transparent; }}
            #chatHistory QScrollBar::handle:vertical {{ background: {asst_border}; border-radius: 4px; }}
            #previewBar {{ background: {card}; border: 1px dashed {asst_border}; border-radius: 12px; }}
            #chatInput {{ border: 1px solid {asst_border}; border-radius: 12px; padding: 8px 12px;
                         background: {card}; color: {ink}; font-size: 13px; }}
            #chatInput:focus {{ border: 1px solid #2563eb; }}
            #sendBtn {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2f6fed, stop:1 #1d4ed8);
                       color: white; border: none; border-radius: 12px;
                       padding: 8px 20px; font-weight: 700; font-size: 13px; }}
            #sendBtn:hover {{ background: #1d4ed8; }}
            #sendBtn:disabled {{ background: #93c5fd; color: #eff6ff; }}
            #stopBtn {{ background: transparent; color: {muted}; border: 1px solid {asst_border};
                       border-radius: 12px; padding: 8px 16px; font-size: 13px; }}
            #stopBtn:hover {{ color: #dc2626; border: 1px solid #dc2626; }}
            #attachBtn {{ background: {subtle}; color: {muted}; border: 1px solid {asst_border};
                         border-radius: 12px; font-size: 15px; }}
            #attachBtn:hover {{ color: #2563eb; border: 1px solid #2563eb; }}
            #thumbBtn {{ background: {card}; border: 1px solid {asst_border}; border-radius: 8px; }}
            #thumbBtn:hover {{ border: 1px solid #dc2626; }}
            #applyBtn {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0ca678, stop:1 #059669);
                        color: white; border: none; border-radius: 12px;
                        padding: 8px 14px; font-weight: 700; font-size: 13px; }}
            #applyBtn:hover {{ background: #047857; }}
            #applyBtn:disabled {{ background: {subtle}; color: {muted}; }}
            #expBtn {{ background: {card}; color: {ink}; border: 1px solid {asst_border};
                      border-radius: 12px; padding: 8px 16px; font-size: 13px; font-weight: 600; }}
            #expBtn:hover {{ border: 1px solid #2563eb; color: #2563eb; }}
            #iconBtn {{ background: transparent; border: none; color: {muted}; border-radius: 8px; }}
            #iconBtn:hover {{ background: {subtle}; }}
            #chatSidebar {{ background: {card}; border: 1px solid {asst_border}; border-radius: 16px; }}
            #sidebarTitle {{ color: {ink}; font-size: 13px; font-weight: 800; }}
            #sessionList {{ background: {card}; color: {ink}; border: none; outline: none; }}
            #sessionList::item {{ background: {subtle}; color: {ink}; border-radius: 10px;
                                 padding: 8px 10px; margin: 3px 0px; font-size: 12px; font-weight: 600; }}
            #sessionList::item:selected {{ background: #2563eb; color: white; }}
            #sessionList::item:hover:!selected {{ background: {asst_border}; }}
            #muted {{ color: {muted}; font-size: 12px; }}
            #statusLbl {{ color: {muted}; font-size: 12px; }}
            #statusDot {{ color: #16a34a; font-size: 13px; }}
            QCheckBox {{ color: {muted}; font-size: 12px; spacing: 5px; }}
            QCheckBox::indicator {{ width: 15px; height: 15px; border-radius: 4px; border: 1px solid {asst_border}; background: {card}; }}
            QCheckBox::indicator:checked {{ background: #2563eb; border: 1px solid #2563eb; }}
            """
        )

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
        self._refresh_view()

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
        self._append_html(
            "<div style='text-align:center; margin:10px 16px 2px;'>"
            "<div style='font-size:15px; font-weight:700;'>Design parts with words — or pictures</div>"
            "<div style='font-size:12px; margin-top:4px;'>"
            "Describe a part, attach a reference sketch or photo, and I'll generate "
            "build123d code, validate it, and render it in the 3D view. "
            "Vision is on: each render sends a screenshot back so I can verify it."
            "</div>"
            "<div style='font-size:12px; margin-top:6px;'>"
            "Try: “30×20×10 bracket, 5 mm holes in each corner, 2 mm fillets”"
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
        if dark:
            palette = {
                "user": ("#2f6fed", "#ffffff", "You"),
                "assistant": ("#232f52", "#e8edf7", "Assistant"),
                "system": ("#101728", "#93a0b8", "Notice"),
                "tool": ("#101728", "#93a0b8", "Tool"),
                "error": ("#3f1d1d", "#fca5a5", "Error"),
                "code": ("#0b1120", "#e8edf7", "Code"),
            }
        else:
            palette = {
                "user": ("#2563eb", "#ffffff", "You"),
                "assistant": ("#ffffff", "#0f1f38", "Assistant"),
                "system": ("#eef1f6", "#64748b", "Notice"),
                "tool": ("#f8fafc", "#475569", "Tool"),
                "error": ("#fef2f2", "#b91c1c", "Error"),
                "code": ("#0f172a", "#e2e8f0", "Code"),
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
                bg, fg, "right", radius="16px 16px 4px 16px"
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
                bg, fg, "left", radius="16px 16px 16px 4px"
            )
        return self._bubble_table(
            f"<div style='line-height:1.45;'>{safe}</div>",
            bg, fg, "left", radius="16px 16px 16px 4px"
        )

    def _code_block(self, code, stamp):
        """Render a generated code block with a working Copy link.

        NOTE: QTextBrowser cannot run JavaScript, so Copy is an anchor
        (cq-copy:<n>) handled by anchorClicked -> clipboard.
        """
        dark = self._app_is_dark()
        cbg = "#0b1120" if dark else "#0f172a"
        cfg = "#e8edf7" if dark else "#e2e8f0"
        safe = html.escape(code)
        try:
            snippet_id = len(self._code_snippets)
            self._code_snippets.append(code)
        except Exception:
            snippet_id = 0
        header = (
            f"<table width='100%' cellspacing='0' cellpadding='0'><tr>"
            f"<td align='left' style='font-size:11px;color:#94a3b8;'>"
            f"Generated build123d code</td>"
            f"<td align='right'><a href='cq-copy:{snippet_id}' "
            f"style='font-size:11px;color:#60a5fa;text-decoration:none;'>"
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
            f"<div style='font-size:11px;color:#64748b;margin-top:4px;'>"
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
        kind = {"you": "user", "assistant": "assistant", "system": "system",
                "tool": "tool", "tool result": "tool", "error": "error"}.get(
            who.lower(), "assistant"
        )
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
        try:
            self.status_dot.setStyleSheet(
                f"color: {'#f59e0b' if busy else '#16a34a'};"
            )
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
        if self.worker is None:
            return
        self._busy_dots = (self._busy_dots + 1) % 4
        base = getattr(self, "_busy_base", "Working")
        self.status.setText(f"{base}{'.' * self._busy_dots}")

    def _thinking_block_html(self, thinking_text):
        """Muted reasoning block (shown live during thinking + in transcript)."""
        dark = self._app_is_dark()
        fg = "#93a0b8" if dark else "#64748b"
        edge = "#33436b" if dark else "#cbd5e1"
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

    def _stream_bubble(self):
        """Live assistant bubble, rebuilt from the buffers on every token."""
        dark = self._app_is_dark()
        bg = "#232f52" if dark else "#ffffff"
        fg = "#e8edf7" if dark else "#0f1f38"
        thinking = ""
        if self._think_buf.strip():
            thinking = self._thinking_block_html(self._think_buf)
        body = html.escape(self._stream_buf).replace("\n", "<br/>")
        return self._bubble_table(
            f"<div style='font-size:11px; margin-bottom:4px;'>"
            f"Assistant &gt; live</div>{thinking}"
            f"<div style='line-height:1.45;'>{body}</div>",
            bg, fg, "left", radius="16px 16px 16px 4px",
        )

    def _append_stream_start(self):
        self._stream_buf = ""
        self._think_buf = ""
        self._stream_dirty = False
        self._stream_active = True
        self._doc_parts.append("")  # reserved slot, rewritten per token
        self._refresh_stream_slot()

    def _refresh_stream_slot(self):
        if self._doc_parts:
            self._doc_parts[-1] = self._stream_bubble()
        else:
            self._doc_parts.append(self._stream_bubble())
        self._refresh_view()

    @pyqtSlot(str)
    def on_stream_text(self, delta):
        if not self._stream_active:
            self._append_stream_start()
        self._stream_buf += delta
        self._busy_base = "Writing"
        self._mark_stream_dirty()

    @pyqtSlot(str)
    def on_thinking_text(self, delta):
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
        if self._stream_active and self._stream_dirty:
            self._stream_dirty = False
            self._refresh_stream_slot()
        self._stream_active = False

    # ------------- reference images -------------
    MAX_IMAGES = 3
    IMAGE_SIDE = 1024  # longest side sent to the model (keeps token cost sane)

    @staticmethod
    def _encode_image(path):
        """Load, downscale and JPEG-encode an image file.

        Returns dict(name, bytes, w, h) or an error string on failure.
        """
        img = QImage(path)
        if img.isNull():
            return f"Could not read image: {path}"
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
            return f"Could not encode image: {path}"
        return {
            "name": Path(path).name,
            "bytes": bytes(buf.data()),
            "w": img.width(),
            "h": img.height(),
        }

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
            item = self._encode_image(path)
            if isinstance(item, str):
                self._append("Error", item)
                continue
            self.pending_images.append(item)
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
            self.session_id, self.messages,
            title=self.session_title, model=getattr(self, "_model", ""),
            usage=self.sess_usage, design_code=self.session_design,
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
                        data = ((block.get("source") or {}).get("data") or "")
                        if data:
                            thumbs.append(data)
                    elif block.get("type") == "tool_result":
                        inner = block.get("content", "")
                        if isinstance(inner, list):
                            for sub in inner:
                                if not isinstance(sub, dict):
                                    continue
                                if sub.get("type") == "image":
                                    data = ((sub.get("source") or {}).get("data") or "")
                                    if data:
                                        shot_thumbs.append(data)
                if texts or thumbs:
                    self._append("You", "\n".join(texts) or "(reference images)",
                                 images=thumbs)
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
                        btype, text = getattr(block, "type", ""), getattr(block, "text", "")
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
        if self.worker is not None:
            self.status.setText("Stop the running request first.")
            return
        self._stash_design()
        # Lazy ChatGPT-style chat: no file/index row yet.
        self.session_id = None
        self.session_title = DEFAULT_TITLE
        self.messages = []
        self.last_code = ""
        self.session_design = ""
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
            self._append("System", "New chat started — empty design, nothing rendered yet.")
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
        self.sess_usage = {"in": int(usage.get("in", 0) or 0),
                           "out": int(usage.get("out", 0) or 0)}
        self.last_usage = {"in": 0, "out": 0}
        self._turn_steps = 0
        recap = (f"Resumed “{self.session_title}” — {len(messages)} messages · "
                 f"in/out {self._fmt_tokens(self.sess_usage['in'])}"
                 f"/{self._fmt_tokens(self.sess_usage['out'])} total")
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
        if QMessageBox.question(
            self, "Delete chat", f"Delete “{title}” permanently?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
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
            "Plan mode ON: code is validated but never auto-applied. "
            "Review it, then press “Insert code + Render”. (/plan to toggle)"
            if self.plan_mode else
            "Plan mode OFF: generated code auto-applies again.",
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
            self._append("System", f"Effort set to {level.title()}.\n{self._effort_summary()}")

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

    def _refresh_model_lbl(self):
        try:
            model = getattr(self, "_model", "") or "from .env"
            self.model_lbl.setText(f"Model: {model} · Effort: {self.effort.title()}")
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
        model = getattr(self, "_model", "") or "from .env"
        spec = EFFORTS[self.effort]
        budget = spec["budget"]
        self._append(
            "System",
            "Session status\n"
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
        model = getattr(self, "_model", "") or "not connected yet — send a message first"
        self._append(
            "System",
            f"Active model: {model}\n"
            "The model always comes from MODEL_ID / FIREWORKS_MODEL_ID in .env "
            "(restart the app after changing it). Vision-capable: kimi-k3.",
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
                parts = [b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text"]
                imgs = sum(1 for b in content
                           if isinstance(b, dict) and b.get("type") == "image")
                shots = 0
                for b in content:
                    if not isinstance(b, dict) or b.get("type") != "tool_result":
                        continue
                    inner = b.get("content", "")
                    if isinstance(inner, list):
                        shots += sum(1 for s in inner
                                     if isinstance(s, dict) and s.get("type") == "image")
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
                        name = block.get("name") if isinstance(block, dict) else block.name
                        texts.append(f"_⚙ tool: {name}_")
                if texts:
                    out += ["## Assistant", "", "\n".join(texts), ""]
        return "\n".join(out).strip() + "\n"

    def _cmd_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save chat transcript", f"{self.session_title}.md",
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
            last = (f"{self._fmt_tokens(self.last_usage['in'])}"
                    f"/{self._fmt_tokens(self.last_usage['out'])} last")
            sess = (f"{self._fmt_tokens(self.sess_usage['in'])}"
                    f"/{self._fmt_tokens(self.sess_usage['out'])} total")
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
        if (not prompt and not self.pending_images) or self.worker is not None:
            return
        # Slash commands run locally, never reach the model.
        if prompt.startswith("/") and not self.pending_images:
            self.input.clear()
            self._hist_idx = None
            self._run_slash(prompt)
            return
        # Model comes from .env ONLY (MODEL_ID / FIREWORKS_MODEL_ID).
        # No UI override - that was the source of 404s (stale picker value
        # beating the .env value).
        try:
            client, model = build_client()
        except RuntimeError as e:
            self._append("System", str(e))
            self.status.setText(str(e))
            return
        self._client, self._model = client, model
        self._refresh_provider_pill()
        self._refresh_model_lbl()
        self._ensure_session()
        _has_images = bool(self.pending_images)
        if is_auto_title(self.session_title) and (prompt or _has_images):
            if _has_images and not prompt:
                blocks_for_title = [{"type": "image"},
                                    {"type": "text", "text": "Reference images"}]
                self.session_title = autotitle(
                    [{"role": "user", "content": blocks_for_title}])
            elif prompt:
                self.session_title = autotitle([{"role": "user", "content": prompt}])
            self._save_session()
            self._refresh_session_combo()
        images = list(self.pending_images)
        self.pending_images = []
        self._refresh_previews()
        if images:
            blocks = [self._image_block(item) for item in images]
            blocks.append({"type": "text", "text": prompt or
                           "Model the part shown in the attached reference image(s) in mm."})
            self.messages.append({"role": "user", "content": blocks})
            thumbs = [base64.b64encode(item["bytes"]).decode("ascii") for item in images]
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
        self._kick_llm()

    def _refresh_provider_pill(self):
        """Show which backend will serve the request (fireworks/anthropic)."""
        import os

        base = ""
        try:
            base = (self._client.base_url or "").lower() if self._client else ""
        except Exception:
            base = ""
        label = "fireworks" if "fireworks" in base else (
            "anthropic" if self._client else "no key"
        )
        try:
            self.provider_pill.setText(f"● {label}")
        except Exception:
            pass

    def eventFilter(self, obj, event):
        """Enter sends, Shift+Enter inserts a newline, Up/Down recalls history."""
        if obj is self.input and event.type() == event.KeyPress:
            key = event.key()
            mods = event.modifiers()
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
        self._hist_idx = max(-1, min(len(self.sent_history) - 1,
                                     self._hist_idx + direction))
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
        w, self.worker = self.worker, None
        if w is not None:
            try:
                w.terminate()
                w.wait(2000)
            except Exception:
                pass
            try:
                w.deleteLater()
            except Exception:
                pass
        self._finish_stream_display()
        self._append("System", "Stopped by user.")
        self._set_busy(False)

    def _kick_llm(self):
        self._set_busy(True)
        # s08-style: send a compacted copy (stale thinking dropped, old code
        # and old tool results shortened). self.messages stays complete.
        from ..ai.cad_agent import compact_messages, thinking_params_for

        payload = compact_messages(self.messages)
        thinking = thinking_params_for(self._model, self.effort)
        max_tokens = max_tokens_for_effort(self.effort)
        self._stream_buf = ""
        self._think_buf = ""
        if self.stream_check.isChecked():
            self._append_stream_start()
            self.worker = StreamWorker(
                self._client, self._model, payload, self,
                thinking=thinking, max_tokens=max_tokens,
            )
            self.worker.text_delta.connect(self.on_stream_text)
            self.worker.thinking_delta.connect(self.on_thinking_text)
        else:
            self.worker = LLMWorker(
                self._client, self._model, payload, self,
                thinking=thinking, max_tokens=max_tokens,
            )
        self.worker.finished_ok.connect(self.on_llm_ok)
        self.worker.finished_err.connect(self.on_llm_err)
        self.worker.start()

    @pyqtSlot(object)
    def on_llm_ok(self, resp):
        if self.worker is not None:
            try:
                self.worker.deleteLater()
            except Exception:
                pass
        self.worker = None
        self._finish_stream_display()
        content = getattr(resp, "content", [])
        stop = getattr(resp, "stop_reason", "") or ""
        self._track_usage(resp)
        self.messages.append({"role": "assistant", "content": content})
        tool_calls = [b for b in content if getattr(b, "type", "") == "tool_use"]
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
                self.status.setText("Reply truncated (max_tokens). Try again / split request.")
            else:
                self._append(
                    "System",
                    "The model returned no text. Press Send again to retry."
                    if not streamed_thinking else
                    "Reasoning shown above, but no reply text followed. "
                    "Press Send again to retry.",
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
                    self._append("System", "Plan ready — review the code, then press “Insert code + Render”.")
                elif self.preferences["Auto-apply code"]:
                    self._append("System", "Code applied to the editor and rendered.")
                    self._auto_apply_verdict = self.apply_code(code)
                else:
                    self._append("System", "Code ready — press “Insert code + Render”.")
        if not tool_calls:
            # No tools requested: if model wrote code as text, offer it.
            # Visual feedback: a text-only code reply that was auto-applied
            # still sends a screenshot so the model can self-correct.
            verdict = getattr(self, "_auto_apply_verdict", "")
            self._auto_apply_verdict = ""
            if (verdict and isinstance(verdict, str) and verdict.startswith("OK")
                    and self._vision_enabled()
                    and self.autofix.isChecked() and self.iters_left > 0):
                capture = self._capture_screenshot()
                if isinstance(capture, bytes):
                    self.messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": (
                                "Auto-applied render OK. Screenshot of the current "
                                "3D view is attached. Inspect it against the request "
                                "and reply with fixed code if it looks wrong, "
                                "otherwise confirm it looks correct."
                            )},
                            self._screenshot_image_block(capture),
                        ],
                    })
                    try:
                        thumb = base64.b64encode(capture).decode("ascii")
                        self._append(
                            "System",
                            "Screenshot sent to the model for visual check.",
                            images=(thumb,),
                        )
                    except Exception:
                        pass
                    self.iters_left -= 1
                    self._update_meter()
                    self._kick_llm()
                    return
            if self.last_code:
                self.status.setText("Code ready. Check editor + 3D view.")
            else:
                self.status.setText("Ready.")
            self._set_busy(False)
            return
        if self.iters_left <= 0:
            self._append("System", "Max iterations reached.")
            self._set_busy(False)
            return
        single_shot = not self.autofix.isChecked()
        results = []
        for block in tool_calls:
            self._turn_steps += 1
            arg = self._tool_arg_preview(block.name, dict(block.input or {}))
            out = self.execute_tool(block.name, dict(block.input or {}))
            content = out
            # Visual feedback: after a successful render, attach a 3D
            # screenshot so the model sees its own design and can iterate.
            if (block.name == "apply_and_render"
                    and isinstance(out, str) and out.startswith("OK")
                    and self._vision_enabled()):
                capture = self._capture_screenshot()
                if isinstance(capture, bytes):
                    verdict = (
                        out + "\nScreenshot of the current 3D view is attached. "
                        "Inspect it against the request and iterate if needed."
                    )
                    content = self._tool_result_with_screenshot(verdict, capture)
                    try:
                        thumb = base64.b64encode(capture).decode("ascii")
                        self._append(
                            "System",
                            "Screenshot sent to the model for visual check.",
                            images=(thumb,),
                        )
                    except Exception:
                        pass
                else:
                    # Screenshot failed: keep text-only so the loop continues.
                    content = out + f"\n(Screenshot unavailable: {capture})"
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": content}
            )
            first_line = (out.strip().splitlines() or [""])[0][:160]
            if out.startswith("OK"):
                self._append_step(f"[{self._turn_steps}] {block.name} {arg} → {first_line}", ok=True)
            else:
                self._append_step(f"[{self._turn_steps}] {block.name} {arg} → needs fix", ok=False)
                self._append("Tool result", out[:2000])
        self._update_meter()
        self.messages.append({"role": "user", "content": results})
        self.iters_left -= 1
        if single_shot or self.iters_left <= 0:
            self._append("System", "Tools executed (self-fix off or limit reached).")
            self._set_busy(False)
            return
        self._kick_llm()

    @pyqtSlot(str)
    def on_llm_err(self, msg):
        if self.worker:
            try:
                self.worker.deleteLater()
            except Exception:
                pass
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
        self.status.setText(msg)
        self._set_busy(False)

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

            # render() emits sigTraceback synchronously in this thread.
            dbg.sigTraceback.connect(_cap)
            try:
                dbg.render()
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
        """Crash-safe validation: child process, never in-process.

        Falls back to in-process headless_validate only when the probe
        itself cannot run (Error:) or the user disabled pre-render.
        """
        try:
            enabled = bool(self.preferences["Isolated pre-render"])
        except Exception:
            enabled = True
        if not enabled:
            return headless_validate(code)
        try:
            timeout = int(self.preferences["Pre-render timeout (s)"])
        except Exception:
            timeout = ISOLATED_PROBE_TIMEOUT
        try:
            self.status.setText("Validating geometry in an isolated process…")
        except Exception:
            pass
        verdict = isolated_validate(code, timeout=timeout,
                                    workdir=self._probe_workdir())
        if verdict.startswith("Error: isolated probe could not run"):
            return headless_validate(code)
        return verdict

    def execute_tool(self, name, args):
        try:
            if name == "get_editor_code":
                return self.get_editor_code()
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
                return self.get_examples()
            if name == "export_model":
                return self.export_model(
                    args.get("format", "stl"), args.get("filename", "")
                )
            if name == "spawn_subagent":
                return self._spawn_subagent_tool(args)
            return f"Error: unknown tool {name}"
        except Exception:
            return "Error:\n" + tb.format_exc(limit=5)

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

            def __init__(self, parent, purpose, task, messages, model, effort, image_bytes=None):
                super().__init__(parent)
                self.purpose = purpose
                self.task = task
                self.messages = messages
                self.model = model
                self.effort = effort
                self.image_bytes = image_bytes

            def run(self):
                try:
                    if self.purpose == "code":
                        current = ""
                        try:
                            current = self.parent()._editor_text()
                        except Exception:
                            pass
                        result = run_code_subagent(
                            request=self.task,
                            current_code=current,
                            parent_messages=self.messages,
                            model=self.model,
                            effort=self.effort,
                        )
                    else:
                        result = run_subagent(
                            task=self.task,
                            parent_messages=self.messages,
                            model=self.model,
                            effort=self.effort,
                            image_bytes=self.image_bytes,
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
            self, purpose, task, list(self.messages),
            getattr(self, "_model", None), self.effort, image_bytes=image_bytes,
        )
        result_holder = {"value": None}

        def on_ok(res):
            result_holder["value"] = res
            self.quit_loop = True

        def on_err(msg):
            result_holder["value"] = {"ok": False, "text": msg, "usage": {"in": 0, "out": 0}}
            self.quit_loop = True

        worker.finished_ok.connect(on_ok)
        worker.finished_err.connect(on_err)
        worker.start()

        # Small event-loop spin so the UI stays responsive while we wait.
        # This is called from the GUI thread during a tool call.
        self.quit_loop = False
        while not self.quit_loop:
            QApplication.processEvents()
            QThread.msleep(20)
        worker.wait(5000)

        result = result_holder["value"] or {"ok": False, "text": "No result", "usage": {"in": 0, "out": 0}}
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

    def get_examples(self):
        try:
            main = self._main()
            if main is None:
                return "Error: main window not found."
            ex_dir = main._examples_dir()
            names = sorted(p.stem for p in Path(ex_dir).glob("*.py"))
            return "Examples: " + ", ".join(names) if names else "No examples found."
        except Exception as e:
            return f"Error listing examples: {e}"

    def apply_code(self, code):
        """Set editor text + trigger debugger.render(), capture traceback.

        Pushes the previous editor text as an undo checkpoint first.
        On success the code becomes this session's design (restored on switch).

        Crash-safe: a CRASH/TIMEOUT verdict from the isolated probe refuses
        the render (rendering it in-process would kill the app). Python-level
        tracebacks still go through the normal render path so they can be
        captured and fed back to the model.
        """
        if not code:
            return "Error: empty code"
        probe = self._validate_isolated(code)
        if probe.startswith(("CRASH:", "TIMEOUT:")):
            self._append("Error", probe)
            try:
                self.status.setText("Render refused: isolated probe crashed.")
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
                return "OK: code applied and rendered, no traceback." + suffix
            return "Rendered WITH traceback:\n" + msg
        except Exception:
            return "Error applying code:\n" + tb.format_exc(limit=5)

    def _append_success_card(self, code):
        """Show a green status card when the model renders successfully."""
        dark = self._app_is_dark()
        bg = "#064e3b" if dark else "#dcfce7"
        fg = "#d1fae5" if dark else "#14532d"
        border = "#10b981" if dark else "#86efac"
        lines = len(code.splitlines())
        chars = len(code)
        stamp = self._now()
        card = (
            f"<div style='margin:8px 2px 12px;'>"
            f"<table width='100%' cellspacing='0' cellpadding='0'><tr><td align='center'>"
            f"<table width='92%' cellspacing='0' cellpadding='12' "
            f"style='background:{bg};border:1px solid {border};border-radius:12px;'>"
            f"<tr><td>"
            f"<div style='font-size:14px;font-weight:800;color:{fg};margin-bottom:4px;'>"
            f"Model created successfully!</div>"
            f"<div style='font-size:12px;color:{fg};line-height:1.45;'>"
            f"Your build123d model was generated and rendered in the 3D view. "
            f"({lines} lines, {chars} chars) · {stamp}"
            f"</div></td></tr></table></td></tr></table></div>"
        )
        self._append_html(card)

    @pyqtSlot()
    def on_apply_last(self):
        if self.last_code:
            out = self.apply_code(self.last_code)
            self._append("System", out)

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

    def export_model(self, fmt="stl", filename=""):
        """Agent-callable export. Defaults to exports/model.<ext> in cwd."""
        fmt = (fmt or "stl").lower().lstrip(".")
        if fmt not in ("stl", "step", "brep"):
            return f"Error: unsupported format '{fmt}'. Use stl/step/brep."
        try:
            shapes = self._collect_shapes()
            if not shapes:
                return "Error: no models to export. Render code first."
            if not filename:
                out_dir = Path.cwd() / "exports"
                out_dir.mkdir(parents=True, exist_ok=True)
                filename = str(out_dir / f"model.{fmt}")
            from ..cq_utils import export as b3d_export

            b3d_export(shapes, fmt, filename)
            return f"OK: exported {len(shapes)} object(s) to {filename}"
        except Exception:
            return "Error exporting:\n" + tb.format_exc(limit=5)

    @pyqtSlot()
    def on_export_button(self, fmt):
        shapes = self._collect_shapes()
        if not shapes:
            self._append("System", "Nothing to export — render a model first.")
            return
        fname = get_save_filename(fmt)
        if not fname:
            return
        try:
            from ..cq_utils import export as b3d_export

            b3d_export(shapes, fmt, fname)
            self._append("System", f"Exported {len(shapes)} object(s) to {fname}")
            self.status.setText(f"Exported to {fname}")
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
                self.sess_usage = {"in": int(usage.get("in", 0) or 0),
                                   "out": int(usage.get("out", 0) or 0)}
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

