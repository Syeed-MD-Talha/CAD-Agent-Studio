"""API profile editor, opened for the provider selected in AI Chat."""

from PyQt5.QtCore import Qt, QPointF, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPalette, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMessageBox,
    QStyledItemDelegate,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..ai.providers import PROVIDERS, ProviderConfig
from ..ai.model_catalog import list_models
from ..theme import theme_for
from .controls import CheckBox


class ProviderComboBox(QComboBox):
    """A padded popup and a crisp chevron independent of the native theme."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(36)
        self.setCursor(Qt.PointingHandCursor)
        self.setView(QListView(self))
        self.setItemDelegate(QStyledItemDelegate(self))
        self.view().setTextElideMode(Qt.ElideNone)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(13)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = theme_for(bool(self.property("dark"))).muted
        painter.setPen(
            QPen(QColor(color), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        )
        x, y = self.width() - 17, self.height() / 2
        painter.drawLine(QPointF(x - 3, y - 1.5), QPointF(x, y + 1.5))
        painter.drawLine(QPointF(x, y + 1.5), QPointF(x + 3, y - 1.5))

    def showPopup(self):
        # Windows can constrain the popup to the closed control's width.
        # Include the delegate's text/icon/padding width before Qt lays it out.
        view = self.view()
        view.ensurePolished()
        content_width = view.sizeHintForColumn(0) + 2 * view.frameWidth() + 24
        view.setMinimumWidth(max(self.width(), content_width))
        super().showPopup()


class ModelListWorker(QThread):
    loaded = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, config):
        # Outlive a closed dialog until the bounded request finishes. Qt
        # disconnects its slots when the dialog is destroyed.
        super().__init__(QApplication.instance())
        self.config = config
        self.finished.connect(self.deleteLater)

    def run(self):
        try:
            models = list_models(self.config, cancelled=self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.loaded.emit(models)
        except Exception as error:
            if not self.isInterruptionRequested():
                message = (
                    str(error)
                    if isinstance(error, RuntimeError)
                    else "Could not load models. Check the API key and base URL."
                )
                self.failed.emit(message.replace(self.config.api_key, "[redacted]"))


class ProviderSettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.provider = config.provider
        self._model_worker = None
        self._catalog_credentials = None
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.setInterval(700)
        self._load_timer.timeout.connect(self._auto_load_models)
        spec = PROVIDERS[self.provider]
        self.setWindowTitle(f"{spec.label} API settings")
        self.setObjectName("providerSettingsDialog")
        self.setMinimumWidth(480)
        self.resize(530, 470)
        self.dark = (
            parent._app_is_dark()
            if hasattr(parent, "_app_is_dark")
            else self.palette().color(QPalette.Window).lightness() < 128
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(20)
        header = QHBoxLayout()
        header.setSpacing(14)
        badge = QLabel(
            {"fireworks": "F", "featherless": "F", "anthropic": "A"}.get(
                self.provider, "+"
            )
        )
        badge.setObjectName("providerBadge")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(46, 46)
        header.addWidget(badge)
        headings = QVBoxLayout()
        headings.setSpacing(4)
        title = QLabel(spec.label)
        title.setObjectName("settingsTitle")
        headings.addWidget(title)
        subtitle = QLabel("Your API key. Your choice of model.")
        subtitle.setObjectName("settingsSubtitle")
        headings.addWidget(subtitle)
        header.addLayout(headings, 1)
        layout.addLayout(header)

        fields = QFrame(self)
        fields.setObjectName("settingsFields")
        form = QVBoxLayout(fields)
        form.setContentsMargins(18, 18, 18, 18)
        form.setSpacing(8)
        key_heading = QHBoxLayout()
        key_label = QLabel("API key")
        key_label.setObjectName("fieldLabel")
        key_heading.addWidget(key_label)
        key_heading.addStretch()
        self.show_key = CheckBox("Show key")
        key_heading.addWidget(self.show_key)
        form.addLayout(key_heading)
        self.api_key = QLineEdit(config.api_key)
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("Paste your API key")
        self.api_key.setAccessibleName("API key")
        self.api_key.setMinimumHeight(42)
        key_label.setBuddy(self.api_key)
        form.addWidget(self.api_key)
        self.show_key.toggled.connect(
            lambda show: self.api_key.setEchoMode(
                QLineEdit.Normal if show else QLineEdit.Password
            )
        )
        form.addSpacing(10)
        model_label = QLabel("Model")
        model_label.setObjectName("fieldLabel")
        model_heading = QHBoxLayout()
        model_heading.addWidget(model_label)
        model_heading.addStretch()
        self.refresh_models_btn = QPushButton("Refresh")
        self.refresh_models_btn.setObjectName("refreshModelsBtn")
        self.refresh_models_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_models_btn.setToolTip(
            "Load the models offered by this API key and endpoint"
        )
        self.refresh_models_btn.setAutoDefault(False)
        self.refresh_models_btn.setMinimumHeight(28)
        self.refresh_models_btn.clicked.connect(self._load_models)
        model_heading.addWidget(self.refresh_models_btn)
        form.addLayout(model_heading)
        self.model = ProviderComboBox()
        self.model.setObjectName("settingsModel")
        self.model.setProperty("dark", self.dark)
        self.model.setEditable(True)
        defaults = [spec.default_model] if spec.default_model else []
        self.model.addItems(
            list(dict.fromkeys(([config.model] if config.model else []) + defaults))
        )
        self.model.setCurrentText(config.model)
        self.model.setInsertPolicy(QComboBox.NoInsert)
        self.model.setMaxVisibleItems(10)
        self._model_completer = QCompleter(self.model.model(), self.model)
        self._model_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._model_completer.setFilterMode(Qt.MatchContains)
        self._model_completer.setCompletionMode(QCompleter.PopupCompletion)
        self._model_completer.setMaxVisibleItems(10)
        self._model_completer.popup().setItemDelegate(
            QStyledItemDelegate(self._model_completer)
        )
        self.model.setCompleter(self._model_completer)
        self.model.setMinimumHeight(42)
        self.model.setAccessibleName("Model ID")
        self.model.lineEdit().setPlaceholderText("Search models or paste a model ID")
        self.model.setToolTip("Paste any model ID available in your provider account.")
        model_label.setBuddy(self.model)
        form.addWidget(self.model)
        hint = QLabel("Choose a model with tool support for CAD generation.")
        hint.setObjectName("fieldHint")
        hint.setWordWrap(True)
        form.addWidget(hint)
        self.model_status = QLabel("Refresh to load models from your provider.")
        self.model_status.setObjectName("modelStatus")
        self.model_status.setWordWrap(True)
        form.addWidget(self.model_status)
        form.addSpacing(6)

        self.advanced_toggle = QToolButton(self)
        self.advanced_toggle.setObjectName("advancedToggle")
        self.advanced_toggle.setText("Advanced settings")
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setCursor(Qt.PointingHandCursor)
        form.addWidget(self.advanced_toggle)
        self.advanced_fields = QWidget(self)
        advanced_layout = QVBoxLayout(self.advanced_fields)
        advanced_layout.setContentsMargins(0, 6, 0, 0)
        advanced_layout.setSpacing(8)
        base_label = QLabel("API base URL")
        base_label.setObjectName("fieldLabel")
        advanced_layout.addWidget(base_label)
        self.base_url = QLineEdit(config.base_url)
        self.base_url.setPlaceholderText("https://your-provider.example/v1")
        self.base_url.setMinimumHeight(42)
        self.base_url.setAccessibleName("API base URL")
        base_label.setBuddy(self.base_url)
        advanced_layout.addWidget(self.base_url)
        form.addWidget(self.advanced_fields)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        expanded = (
            self.provider.startswith("custom_")
            or config.base_url != spec.default_base_url
        )
        self.advanced_toggle.setChecked(expanded)
        self._toggle_advanced(expanded)
        layout.addWidget(fields)

        privacy = QLabel(
            "Saved on this device, unencrypted. Never included in chat exports."
        )
        privacy.setObjectName("privacyNote")
        privacy.setWordWrap(True)
        layout.addWidget(privacy)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons = buttons
        buttons.button(QDialogButtonBox.Save).setText("Save settings")
        buttons.button(QDialogButtonBox.Save).setObjectName("saveProviderBtn")
        buttons.button(QDialogButtonBox.Save).setDefault(True)
        for button in buttons.buttons():
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(38)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._apply_styles()
        self.api_key.textEdited.connect(self._catalog_inputs_changed)
        self.base_url.textEdited.connect(self._catalog_inputs_changed)

    def showEvent(self, event):
        super().showEvent(event)
        self._load_timer.start(0)

    def done(self, result):
        self._load_timer.stop()
        if self._model_worker is not None:
            self._model_worker.requestInterruption()
            self._model_worker = None
        super().done(result)

    def _catalog_inputs_changed(self):
        if self._model_worker is not None:
            self._model_worker.requestInterruption()
            self._model_worker = None
        self._catalog_credentials = None
        current = self.model.currentText()
        self.model.clear()
        if current:
            self.model.addItem(current)
        self._set_model_loading(False)
        self._set_model_status("API settings changed. Refresh to load matching models.")
        if self.isVisible():
            self._load_timer.start(700)

    def _auto_load_models(self):
        if (
            self.isVisible()
            and self.api_key.text().strip()
            and self.base_url.text().strip()
        ):
            credentials = (self.api_key.text().strip(), self.base_url.text().strip())
            if credentials != self._catalog_credentials:
                self._load_models()

    def _set_model_status(self, text, error=False):
        self.model_status.setText(text)
        color = (
            ("#f87171" if self.dark else "#dc2626")
            if error
            else theme_for(self.dark).muted
        )
        self.model_status.setStyleSheet(f"color: {color}; font-size: 11px;")

    def _set_model_loading(self, loading):
        self.refresh_models_btn.setEnabled(not loading)
        self.refresh_models_btn.setText("Loading…" if loading else "Refresh")

    def _load_models(self):
        self._load_timer.stop()
        if self._model_worker is not None:
            return
        config = self.config()
        if not config.api_key or not config.base_url:
            self._set_model_status(
                "Enter your API key and base URL, then click Refresh.", error=True
            )
            return
        try:
            config.validate(require_model=False)
        except RuntimeError as error:
            self._set_model_status(str(error), error=True)
            return
        self._set_model_loading(True)
        self._set_model_status("Loading models from your provider…")
        worker = ModelListWorker(config)
        self._model_worker = worker
        worker.loaded.connect(self._models_loaded)
        worker.failed.connect(self._models_failed)
        worker.start()

    def _current_model_request(self):
        worker = self.sender()
        if worker is not self._model_worker:
            return False
        credentials = (self.api_key.text().strip(), self.base_url.text().strip())
        if credentials != (worker.config.api_key, worker.config.base_url):
            self._catalog_inputs_changed()
            return False
        self._catalog_credentials = credentials
        self._model_worker = None
        self._set_model_loading(False)
        return True

    def _models_loaded(self, models):
        if not self._current_model_request():
            return
        current = self.model.currentText().strip()
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItems(models)
        for index, model_id in enumerate(models):
            self.model.setItemData(index, model_id, Qt.ToolTipRole)
        self.model.setCurrentText(current or (models[0] if models else ""))
        self.model.blockSignals(False)
        self._set_model_status(
            f"{len(models)} models loaded. Type to search, or open the list."
            if models
            else "No models returned. You can still paste a model ID."
        )

    def _models_failed(self, message):
        if self._current_model_request():
            self._set_model_status(message, error=True)

    def _toggle_advanced(self, expanded):
        self.advanced_fields.setVisible(expanded)
        self.advanced_toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def _apply_styles(self):
        colors = theme_for(self.dark)
        bg, card, ink, muted, border = (
            colors.window,
            colors.surface,
            colors.text,
            colors.muted,
            colors.border,
        )
        accent_bg, accent_ink = colors.selection, colors.accent
        self.setStyleSheet(f"""
            QDialog#providerSettingsDialog {{ background: {bg}; }}
            QLabel {{ color: {ink}; background: transparent; border: none; }}
            #providerBadge {{ background: {accent_bg}; color: {accent_ink}; border-radius: 14px;
                              font-size: 23px; font-weight: 700; }}
            #settingsTitle {{ font-size: 20px; font-weight: 700; }}
            #settingsSubtitle, #privacyNote {{ color: {muted}; font-size: 12px; }}
            #settingsFields {{ background: {card}; border: 1px solid {border}; border-radius: 14px; }}
            #fieldLabel {{ font-size: 12px; font-weight: 600; }}
            #fieldHint, #modelStatus {{ color: {muted}; font-size: 11px; }}
            QLineEdit, QComboBox {{ background: {card}; color: {ink}; border: 1px solid {border};
                border-radius: 8px; padding: 0px 12px; font-size: 13px;
                selection-background-color: {colors.primary}; selection-color: white; }}
            QLineEdit:hover, QComboBox:hover {{ border-color: {accent_ink}; }}
            QLineEdit:focus, QComboBox:focus {{ border: 1px solid {accent_ink}; }}
            QComboBox {{ padding-right: 30px; }}
            QComboBox QLineEdit {{ border: none; background: transparent; padding: 0px; border-radius: 0px; }}
            QComboBox::drop-down {{ border: none; width: 28px; }}
            QComboBox::down-arrow {{ image: none; }}
            QComboBox QAbstractItemView {{ background: {card}; color: {ink}; border: 1px solid {border};
                padding: 6px; outline: none; selection-background-color: {accent_bg}; selection-color: {ink}; }}
            QComboBox QAbstractItemView::item {{ min-height: 32px; padding: 2px 8px; }}
            QCheckBox {{ color: {muted}; font-size: 12px; spacing: 8px; }}
            QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px;
                border: 1px solid {border}; background: {card}; }}
            QCheckBox::indicator:hover, QCheckBox::indicator:focus {{ border-color: {accent_ink}; }}
            QCheckBox::indicator:checked {{ background: {colors.primary}; border-color: {colors.primary}; }}
            QCheckBox::indicator:checked:hover {{ background: {colors.primary_hover}; border-color: {colors.primary_hover}; }}
            QCheckBox::indicator:checked:disabled {{ background: {muted}; border-color: {muted}; }}
            #advancedToggle {{ background: transparent; color: {muted}; border: none;
                               padding: 4px 0px; font-size: 12px; }}
            #advancedToggle:hover {{ color: {accent_ink}; }}
            QPushButton {{ background: {card}; color: {ink}; border: 1px solid {border};
                border-radius: 9px; padding: 0px 18px; font-size: 12px; font-weight: 600; }}
            QPushButton:hover {{ background: {accent_bg}; border-color: {accent_ink}; color: {accent_ink}; }}
            QPushButton:pressed {{ background: {colors.subtle}; border-color: {accent_ink}; }}
            QPushButton:focus {{ border-color: {accent_ink}; }}
            QPushButton:disabled {{ background: {colors.subtle}; color: {muted}; border-color: {border}; }}
            #refreshModelsBtn {{ color: {accent_ink}; background: {accent_bg}; border: none;
                                 border-radius: 6px; padding: 0px 10px; font-size: 11px; }}
            #refreshModelsBtn:hover {{ background: {colors.subtle}; }}
            #refreshModelsBtn:pressed {{ background: {border}; }}
            #refreshModelsBtn:disabled {{ color: {muted}; background: {colors.subtle}; }}
            #saveProviderBtn {{ background: {colors.primary}; color: white; border: 1px solid {colors.primary}; }}
            #saveProviderBtn:hover {{ background: {colors.primary_hover}; border-color: {colors.primary_hover}; }}
            #saveProviderBtn:pressed {{ background: {colors.primary_pressed}; border-color: {colors.primary_pressed}; }}
            #saveProviderBtn:focus {{ border-color: {ink}; }}
            #saveProviderBtn:disabled {{ background: {colors.subtle}; color: {muted}; border-color: {border}; }}
        """)
        # Qt can retain the ancestor chat's native control styling on dialogs.
        # Give interactive controls the dialog rules directly as well.
        for control in (
            self.model,
            self.api_key,
            self.base_url,
            self.refresh_models_btn,
        ):
            control.setStyleSheet(self.styleSheet())
        self._model_completer.popup().setStyleSheet(f"""
            QAbstractItemView {{ background: {card}; color: {ink}; border: 1px solid {border};
                padding: 0px; outline: none; selection-background-color: {accent_bg}; selection-color: {ink}; }}
            QAbstractItemView::item {{ min-height: 32px; padding: 2px 8px; }}
        """)
        for button in self.buttons.buttons():
            button.setStyleSheet(self.styleSheet())

    def config(self):
        return ProviderConfig(
            self.provider,
            self.api_key.text().strip(),
            self.model.currentText().strip(),
            self.base_url.text().strip(),
        )

    def accept(self):
        try:
            self.config().validate()
        except RuntimeError as error:
            QMessageBox.warning(self, "Provider settings", str(error))
            return
        super().accept()
