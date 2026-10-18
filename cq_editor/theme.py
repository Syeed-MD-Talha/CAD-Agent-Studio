"""Shared colors and desktop chrome, leaving code-editor palettes independent."""

from dataclasses import dataclass
import os
import sys
from pathlib import Path

from PyQt5.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt5.QtWidgets import QApplication


@dataclass(frozen=True)
class Theme:
    dark: bool
    window: str
    surface: str
    subtle: str
    text: str
    muted: str
    border: str
    accent: str
    selection: str
    viewport: str
    primary: str
    primary_hover: str
    primary_pressed: str


LIGHT = Theme(
    False,
    "#f4efe5",
    "#fffdf7",
    "#eee8dc",
    "#302f2a",
    "#756f63",
    "#ded6c7",
    "#416653",
    "#e5ecdf",
    "#f5f1e8",
    "#416653",
    "#355644",
    "#294535",
)
DARK = Theme(
    True,
    "#0d1422",
    "#151f30",
    "#1d2b40",
    "#e6edf7",
    "#99a9bf",
    "#2b3b53",
    "#6b9bff",
    "#253c63",
    "#101a2a",
    "#2563eb",
    "#1d4ed8",
    "#1e40af",
)


def theme_for(dark=False):
    return DARK if dark else LIGHT


def palette_for(theme):
    palette = QPalette()
    for role, color in (
        (QPalette.Window, theme.window),
        (QPalette.WindowText, theme.text),
        (QPalette.Base, theme.surface),
        (QPalette.AlternateBase, theme.subtle),
        (QPalette.Text, theme.text),
        (QPalette.Button, theme.surface),
        (QPalette.ButtonText, theme.text),
        (QPalette.ToolTipBase, theme.surface),
        (QPalette.ToolTipText, theme.text),
        (QPalette.Highlight, theme.accent),
        (QPalette.HighlightedText, "#ffffff"),
        (QPalette.Link, theme.accent),
        (QPalette.BrightText, "#ef4444"),
        (QPalette.Mid, theme.border),
        (QPalette.PlaceholderText, theme.muted),
    ):
        palette.setColor(role, QColor(color))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor(theme.muted))
    return palette


def apply_application_theme(dark=False):
    app = QApplication.instance()
    theme = theme_for(dark)
    if not app.property("studioStyleInitialized"):
        app.setStyle("Fusion")
        app.setProperty("studioStyleInitialized", True)
    if not app.property("studioFontInitialized"):
        # Some Windows Qt installations do not discover the system font folder.
        # Register the installed fonts explicitly; no font files are bundled.
        if sys.platform == "win32":
            font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
            for filename in (
                "segoeui.ttf",
                "seguisb.ttf",
                "segoeuib.ttf",
                "segoeuii.ttf",
                "consola.ttf",
            ):
                font_path = font_dir / filename
                if font_path.is_file():
                    QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase().families()
        family = next(
            (name for name in ("Segoe UI", "Inter", "Noto Sans") if name in families),
            app.font().family(),
        )
        app.setFont(QFont(family, 10))
        app.setProperty("studioFontInitialized", True)
    app.setPalette(palette_for(theme))
    # No QWidget/QPlainTextEdit rule: syntax colors and editor preferences
    # remain in control of the editor, rather than being overridden by QSS.
    app.setStyleSheet(f"""
        QToolTip {{ background: {theme.surface}; color: {theme.text}; border: 1px solid {theme.border}; padding: 6px 9px; }}
        QMainWindow::separator {{ width: 7px; height: 7px; background: {theme.window}; }}
        QMainWindow::separator:hover {{ background: {theme.selection}; }}
        QMenuBar {{ background: {theme.surface}; color: {theme.muted}; border-bottom: 1px solid {theme.border}; padding: 4px 12px; }}
        QMenuBar::item {{ background: transparent; padding: 5px 10px; border-radius: 5px; }}
        QMenuBar::item:selected {{ background: {theme.selection}; color: {theme.text}; }}
        QMenu {{ background: {theme.surface}; color: {theme.text}; border: 1px solid {theme.border}; padding: 6px; }}
        QMenu::item {{ padding: 7px 30px 7px 24px; border-radius: 5px; }}
        QMenu::item:selected {{ background: {theme.selection}; }}
        QMenu::item:disabled {{ color: {theme.muted}; }}
        QMenu::separator {{ height: 1px; background: {theme.border}; margin: 5px 8px; }}
        QToolBar {{ background: {theme.surface}; border: none; border-bottom: 1px solid {theme.border}; spacing: 5px; padding: 8px 14px; }}
        QToolBar::separator {{ background: {theme.border}; width: 1px; margin: 7px 9px; }}
        QToolBar QComboBox {{ color: {theme.text}; background: {theme.surface}; border: 1px solid {theme.border};
            border-radius: 8px; padding: 3px 27px 3px 10px; font-weight: 600; }}
        QToolBar QComboBox:hover {{ background: {theme.selection}; border-color: {theme.accent}; }}
        QToolBar QComboBox::drop-down {{ border: none; width: 25px; }}
        QToolBar QComboBox::down-arrow {{ image: none; }}
        QToolButton {{ color: {theme.text}; background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 6px; }}
        QToolButton:hover {{ background: {theme.selection}; border-color: {theme.border}; }}
        QToolButton:pressed {{ background: {theme.subtle}; border-color: {theme.accent}; }}
        QToolButton:checked {{ background: {theme.selection}; color: {theme.accent}; }}
        QToolButton:disabled {{ color: {theme.muted}; }}
        QToolButton:focus {{ border-color: {theme.accent}; }}
        QToolButton::menu-indicator {{ subcontrol-position: right bottom; }}
        QToolBar QToolButton {{ background: {theme.surface}; border: 1px solid {theme.border};
            border-radius: 9px; padding: 8px 10px; }}
        QToolBar QToolButton:hover {{ background: {theme.selection}; border-color: {theme.accent}; }}
        QToolBar QToolButton:pressed {{ background: {theme.subtle}; border-color: {theme.accent}; }}
        QToolButton#renderAction {{ background: {theme.primary}; border-color: {theme.primary};
            color: white; padding: 8px 18px; font-weight: 600; }}
        QToolButton#renderAction:hover {{ background: {theme.primary_hover}; border-color: {theme.primary_hover}; }}
        QToolButton#renderAction:pressed {{ background: {theme.primary_pressed}; border-color: {theme.primary_pressed}; }}
        QToolButton#renderAction:disabled {{ background: {theme.subtle}; border-color: {theme.border}; color: {theme.muted}; }}
        QPushButton {{ background: {theme.surface}; color: {theme.text}; border: 1px solid {theme.border};
            border-radius: 8px; padding: 6px 12px; }}
        QPushButton:hover {{ background: {theme.selection}; border-color: {theme.accent}; }}
        QPushButton:pressed {{ background: {theme.subtle}; border-color: {theme.accent}; }}
        QPushButton:focus {{ border-color: {theme.accent}; }}
        QPushButton:disabled {{ background: {theme.subtle}; color: {theme.muted}; border-color: {theme.border}; }}
        QLabel#studioBrand {{ color: {theme.text}; font-size: 15px; font-weight: 700; padding-right: 12px; }}
        QLabel#studioBadge {{ color: {theme.muted}; font-size: 11px; padding: 4px 10px; }}
        QDockWidget {{ color: {theme.text}; border: 1px solid {theme.border}; }}
        QTabBar::tab {{ background: {theme.window}; color: {theme.muted}; padding: 8px 14px;
            border: none; border-bottom: 2px solid transparent; }}
        QTabBar::tab:selected {{ background: {theme.surface}; color: {theme.text}; border-bottom-color: {theme.accent}; }}
        QTabBar::tab:hover {{ color: {theme.accent}; }}
        QTreeView, QTableView {{ background: {theme.surface}; alternate-background-color: {theme.subtle};
            color: {theme.text}; border: none; outline: none; selection-background-color: {theme.selection};
            selection-color: {theme.text}; }}
        QTreeView::item {{ padding: 5px 4px; border: none; }}
        QTreeView::item:selected, QTableView::item:selected {{ background: {theme.selection}; color: {theme.text}; }}
        QTreeView::item:hover {{ background: {theme.subtle}; }}
        QHeaderView::section {{ background: {theme.surface}; color: {theme.muted}; border: none;
            border-bottom: 1px solid {theme.border}; padding: 7px 10px; font-weight: 600; }}
        QTableView {{ gridline-color: {theme.border}; }}
        QStatusBar {{ background: {theme.surface}; color: {theme.muted}; border-top: 1px solid {theme.border}; }}
        QStatusBar::item {{ border: none; }}
        QStatusBar QLabel {{ color: {theme.muted}; padding: 3px 8px; font-size: 11px; }}
        QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: {theme.border}; min-height: 24px; border-radius: 4px; }}
        QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
        QScrollBar::handle:horizontal {{ background: {theme.border}; min-width: 24px; border-radius: 4px; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
        QSplitter::handle {{ background: {theme.window}; }}
        QSplitter::handle:hover {{ background: {theme.selection}; }}
    """)
    return theme


def dialog_styles(theme):
    """Shared form styling for preferences and export/upload dialogs."""
    return f"""
        QDialog {{ background: {theme.window}; color: {theme.text}; }}
        QLabel {{ color: {theme.text}; }}
        QLabel#dialogHeading {{ font-size: 21px; font-weight: 700; }}
        QLabel#dialogDescription {{ color: {theme.muted}; font-size: 12px; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: {theme.surface}; color: {theme.text};
            border: 1px solid {theme.border}; border-radius: 8px; padding: 7px 10px; min-height: 20px; }}
        QLineEdit:focus, QComboBox:focus {{ border-color: {theme.accent}; }}
        QPushButton {{ background: {theme.surface}; color: {theme.text}; border: 1px solid {theme.border};
            border-radius: 9px; padding: 9px 18px; font-weight: 600; }}
        QPushButton:hover {{ background: {theme.selection}; border-color: {theme.accent}; }}
        QPushButton:pressed {{ background: {theme.subtle}; border-color: {theme.accent}; }}
        QPushButton:focus {{ border-color: {theme.accent}; }}
        QPushButton:disabled {{ background: {theme.subtle}; color: {theme.muted}; border-color: {theme.border}; }}
        QPushButton#primaryAction {{ background: {theme.primary}; color: white; border-color: {theme.primary}; }}
        QPushButton#primaryAction:hover {{ background: {theme.primary_hover}; border-color: {theme.primary_hover}; }}
        QPushButton#primaryAction:pressed {{ background: {theme.primary_pressed}; border-color: {theme.primary_pressed}; }}
        QPushButton#primaryAction:disabled {{ background: {theme.subtle}; color: {theme.muted}; border-color: {theme.border}; }}
        QTextBrowser {{ background: {theme.surface}; color: {theme.text}; border: 1px solid {theme.border}; border-radius: 8px; padding: 10px; }}
        QTreeWidget#preferencesNavigation {{ background: {theme.surface}; border: 1px solid {theme.border};
            border-radius: 10px; padding: 8px; }}
        QTreeWidget#preferencesNavigation::item {{ padding: 10px 8px; border-radius: 6px; }}
        QTreeWidget#preferencesNavigation::item:selected {{ background: {theme.selection}; color: {theme.accent}; }}
    """
