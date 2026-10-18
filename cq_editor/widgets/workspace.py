"""Desktop workspace chrome around the existing dock panels and OCCT canvas."""

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
import qtawesome as qta
from .controls import ElidedLabel


class DockHeader(QWidget):
    def __init__(self, dock):
        super().__init__(dock)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.dock = dock
        self.setObjectName("dockHeader")
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 5, 7, 5)
        row.setSpacing(7)
        self.label = QLabel(dock.windowTitle(), self)
        row.addWidget(self.label, 1)
        self.float_btn = QToolButton(self)
        self.float_btn.setToolTip("Float or dock this panel")
        self.float_btn.setAccessibleName("Float or dock panel")
        self.float_btn.clicked.connect(lambda: dock.setFloating(not dock.isFloating()))
        self.close_btn = QToolButton(self)
        self.close_btn.setToolTip("Hide panel · reopen from Workspace")
        self.close_btn.setAccessibleName("Hide panel")
        self.close_btn.clicked.connect(dock.close)
        for button in (self.float_btn, self.close_btn):
            button.setFixedSize(24, 24)
            button.setCursor(Qt.PointingHandCursor)
            button.setIconSize(QSize(11, 11))
            row.addWidget(button)
        dock.windowTitleChanged.connect(self.label.setText)

    def apply_theme(self, theme):
        self.setStyleSheet(f"""
            #dockHeader {{ background: {theme.surface}; border-bottom: 1px solid {theme.border}; }}
            QLabel {{ color: {theme.muted}; font-size: 11px; font-weight: 600; }}
            QToolButton {{ border: none; border-radius: 5px; padding: 3px; background: transparent; }}
            QToolButton:hover {{ background: {theme.subtle}; }}
            QToolButton:pressed {{ background: {theme.selection}; }}
            QToolButton:focus {{ border: 1px solid {theme.accent}; }}
        """)
        self.float_btn.setIcon(qta.icon("fa5.clone", color=theme.muted))
        self.close_btn.setIcon(qta.icon("fa5s.times", color=theme.muted))


class ViewportPanel(QFrame):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.setObjectName("viewportPanel")
        self.setMinimumSize(260, 240)
        self.viewer = viewer
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QWidget(self)
        header.setObjectName("viewportHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(14, 8, 10, 8)
        title = QLabel("3D viewport", header)
        title.setObjectName("viewportTitle")
        row.addWidget(title)
        row.addStretch()
        actions = viewer._actions["View"]
        self.fit_btn = QToolButton(header)
        self.fit_btn.setDefaultAction(actions[0])
        self.fit_btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.iso_btn = QToolButton(header)
        self.iso_btn.setDefaultAction(actions[1])
        self.views_btn = QToolButton(header)
        self.views_btn.setText("Views")
        self.views_btn.setPopupMode(QToolButton.InstantPopup)
        self.views_btn.setMenu(QMenu(self.views_btn))
        self.views_btn.menu().addActions(actions)
        self.views_btn.menu().addSeparator()
        self.views_btn.menu().addActions(viewer._actions["Tools"])
        for button in (self.fit_btn, self.iso_btn, self.views_btn):
            button.setCursor(Qt.PointingHandCursor)
            row.addWidget(button)
        layout.addWidget(header)
        layout.addWidget(viewer.canvas, 1)
        footer = QWidget(self)
        footer.setObjectName("viewportFooter")
        footer_row = QHBoxLayout(footer)
        footer_row.setContentsMargins(14, 7, 14, 7)
        self.object_count = QLabel("0 objects", footer)
        self.navigation_hint = ElidedLabel(
            "Drag to orbit · Ctrl + drag to pan · Scroll to zoom", footer
        )
        self.navigation_hint.setToolTip(self.navigation_hint.text())
        self.navigation_hint.setAlignment(Qt.AlignCenter)
        self.projection = QLabel("Orthographic", footer)
        viewer.preferences.sigTreeStateChanged.connect(self._refresh_projection)
        footer_row.addWidget(self.object_count)
        footer_row.addWidget(self.navigation_hint, 1)
        footer_row.addWidget(self.projection)
        layout.addWidget(footer)

    def _refresh_projection(self, *_):
        self.projection.setText(self.viewer.preferences["Projection Type"])

    def apply_theme(self, theme):
        self.setStyleSheet(f"""
            #viewportPanel {{ border: 1px solid {theme.border}; background: {theme.viewport}; }}
            #viewportHeader, #viewportFooter {{ background: {theme.surface}; }}
            #viewportHeader {{ border-bottom: 1px solid {theme.border}; }}
            #viewportFooter {{ border-top: 1px solid {theme.border}; }}
            QLabel {{ color: {theme.muted}; font-size: 10px; border: none; }}
            #viewportTitle {{ color: {theme.text}; font-size: 12px; font-weight: 600; }}
            QToolButton {{ color: {theme.text}; background: {theme.surface}; font-size: 11px;
                border: 1px solid {theme.border}; border-radius: 7px; padding: 5px 8px; }}
            QToolButton:hover {{ background: {theme.selection}; border-color: {theme.accent}; }}
            QToolButton:pressed {{ background: {theme.subtle}; border-color: {theme.accent}; }}
            QToolButton:focus {{ border-color: {theme.accent}; }}
        """)
        self.fit_btn.setIcon(qta.icon("fa5s.expand", color=theme.muted))
        self.iso_btn.setIcon(qta.icon("fa5s.cube", color=theme.muted))
