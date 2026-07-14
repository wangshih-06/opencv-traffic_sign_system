"""Modern light and dark QSS themes for the desktop interface."""

from __future__ import annotations

from PyQt5.QtWidgets import QApplication


PRIMARY = "#2F54EB"
PRIMARY_LIGHT = "#5173F5"
PRIMARY_DARK = "#1D39C4"
ACCENT = "#1E88E5"
ACCENT_LIGHT = "#64B5F6"

BG_LIGHT = "#F6F8FC"
PANEL_LIGHT = "#FFFFFF"
BORDER_LIGHT = "#E2E8F0"
TEXT_LIGHT = "#263238"
MUTED_LIGHT = "#7B8794"
INPUT_LIGHT = "#FFFFFF"
DISABLED_BG_LIGHT = "#EEF1F5"

BG_DARK = "#171A23"
PANEL_DARK = "#222633"
BORDER_DARK = "#343A4B"
TEXT_DARK = "#E6EAF2"
INPUT_DARK = "#1B1F2A"
MUTED_DARK = "#929BAA"
DISABLED_BG_DARK = "#303543"


def _build_theme(*, dark: bool) -> str:
    bg = BG_DARK if dark else BG_LIGHT
    panel = PANEL_DARK if dark else PANEL_LIGHT
    border = BORDER_DARK if dark else BORDER_LIGHT
    text = TEXT_DARK if dark else TEXT_LIGHT
    muted = MUTED_DARK if dark else MUTED_LIGHT
    input_bg = INPUT_DARK if dark else INPUT_LIGHT
    disabled = DISABLED_BG_DARK if dark else DISABLED_BG_LIGHT
    soft = "#292E3D" if dark else "#F4F7FB"
    hover_soft = "#303647" if dark else "#EDF3FF"
    status_bg = "#1E222E" if dark else "#FFFFFF"
    danger_text = "#FF7875" if dark else "#D9363E"
    danger_bg = "#39282E" if dark else "#FFF1F0"
    selected_bg = "#35436E" if dark else "#EAF0FF"

    return f"""
QMainWindow, QDialog {{
    background-color: {bg};
    color: {text};
    font-family: "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}}
QWidget {{
    color: {text};
    font-family: "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}}
QWidget#leftPanelContent, QWidget#rightPanel, QWidget#sectionBody {{
    background: transparent;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    border: none;
    background: transparent;
}}
QWidget#sectionCard {{
    background-color: {panel};
    border: 1px solid {border};
    border-radius: 10px;
}}
QToolButton#collapsibleHeader {{
    background: transparent;
    border: none;
    border-bottom: 1px solid {border};
    border-radius: 0;
    color: {text};
    font-weight: 600;
    text-align: left;
    padding: 11px 14px;
}}
QToolButton#collapsibleHeader:hover {{
    background-color: {hover_soft};
    color: {PRIMARY_LIGHT if dark else PRIMARY};
}}
QGroupBox {{
    background-color: {panel};
    border: 1px solid {border};
    border-radius: 10px;
    margin-top: 16px;
    padding-top: 10px;
    font-weight: 600;
    color: {text};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: {PRIMARY_LIGHT if dark else PRIMARY};
    background: {panel};
}}
QGroupBox#resultCard QLabel#resultValue {{
    color: {PRIMARY_LIGHT if dark else PRIMARY};
    font-size: 18px;
    font-weight: 700;
}}
QGroupBox#resultCard QLabel#resultName {{
    color: {text};
    font-size: 15px;
    font-weight: 600;
}}
QPushButton {{
    min-height: 20px;
    border-radius: 7px;
    padding: 8px 15px;
    font-weight: 500;
}}
QPushButton#primaryButton {{
    background-color: {PRIMARY};
    color: white;
    border: 1px solid {PRIMARY};
}}
QPushButton#primaryButton:hover {{
    background-color: {PRIMARY_LIGHT};
    border-color: {PRIMARY_LIGHT};
}}
QPushButton#primaryButton:pressed {{
    background-color: {PRIMARY_DARK};
    border-color: {PRIMARY_DARK};
}}
QPushButton#secondaryButton, QPushButton#dangerButton {{
    background-color: {panel};
    color: {text};
    border: 1px solid {border};
}}
QPushButton#secondaryButton:hover {{
    color: {PRIMARY_LIGHT if dark else PRIMARY};
    border-color: {PRIMARY_LIGHT if dark else PRIMARY};
    background-color: {hover_soft};
}}
QPushButton#dangerButton:hover {{
    color: {danger_text};
    border-color: {danger_text};
    background-color: {danger_bg};
}}
QPushButton#modeButton {{
    background-color: {soft};
    color: {muted};
    border: 1px solid {border};
    padding: 8px 18px;
}}
QPushButton#modeButton:hover {{
    color: {PRIMARY_LIGHT if dark else PRIMARY};
    border-color: {PRIMARY_LIGHT if dark else PRIMARY};
}}
QPushButton#modeButton:checked {{
    background-color: {selected_bg};
    color: {PRIMARY_LIGHT if dark else PRIMARY};
    border: 1px solid {PRIMARY_LIGHT if dark else PRIMARY};
    font-weight: 600;
}}
QPushButton:disabled {{
    background-color: {disabled};
    color: {muted};
    border: 1px solid {border};
}}
QLineEdit, QComboBox, QSpinBox {{
    background-color: {input_bg};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 7px 9px;
    min-height: 20px;
    selection-background-color: {PRIMARY};
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover {{
    border-color: {PRIMARY_LIGHT if dark else "#A8B8D8"};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
    border: 1px solid {PRIMARY_LIGHT if dark else PRIMARY};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{
    background-color: {disabled};
    color: {muted};
}}
QProgressBar {{
    background-color: {soft};
    color: {text};
    border: none;
    border-radius: 7px;
    text-align: center;
    font-size: 11px;
}}
QProgressBar::chunk {{
    background-color: {PRIMARY};
    border-radius: 7px;
}}
QProgressBar#top1Bar::chunk {{ background-color: #2F54EB; }}
QProgressBar#top2Bar::chunk {{ background-color: #1E88E5; }}
QProgressBar#top3Bar::chunk {{ background-color: #52C41A; }}
QListWidget {{
    background-color: {input_bg};
    color: {text};
    border: 1px solid {border};
    border-radius: 7px;
    padding: 4px;
    alternate-background-color: {soft};
    outline: none;
}}
QListWidget::item {{
    min-height: 27px;
    padding: 3px 7px;
    border-radius: 5px;
}}
QListWidget::item:selected {{
    background-color: {selected_bg};
    color: {PRIMARY_LIGHT if dark else PRIMARY};
}}
QLabel#imageCanvas {{
    background-color: {soft};
    color: {muted};
    border: 1px dashed {border};
    border-radius: 9px;
    padding: 8px;
}}
QMenuBar {{
    background-color: {panel};
    border-bottom: 1px solid {border};
    padding: 3px 6px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 6px 11px;
    border-radius: 5px;
}}
QMenuBar::item:selected {{
    background-color: {selected_bg};
    color: {PRIMARY_LIGHT if dark else PRIMARY};
}}
QMenu {{
    background-color: {panel};
    color: {text};
    border: 1px solid {border};
    border-radius: 7px;
    padding: 5px;
}}
QMenu::item {{ padding: 7px 24px; border-radius: 4px; }}
QMenu::item:selected {{
    background-color: {selected_bg};
    color: {PRIMARY_LIGHT if dark else PRIMARY};
}}
QStatusBar {{
    background-color: {status_bg};
    border-top: 1px solid {border};
    min-height: 31px;
}}
QStatusBar QLabel#operationStatus {{
    color: {PRIMARY_LIGHT if dark else PRIMARY};
    font-weight: 600;
    padding: 0 12px;
}}
QStatusBar QLabel#statusMetric {{
    color: {muted};
    border-left: 1px solid {border};
    padding: 0 12px;
}}
QToolButton#themeToggle {{
    background: transparent;
    color: {text};
    border: none;
    border-radius: 14px;
    font-size: 16px;
}}
QToolButton#themeToggle:hover {{ background-color: {hover_soft}; }}
QSplitter::handle {{ background-color: {border}; }}
QSplitter::handle:hover {{ background-color: {PRIMARY_LIGHT if dark else PRIMARY}; }}
QScrollBar:vertical {{
    background: transparent;
    width: 9px;
    margin: 3px 1px;
}}
QScrollBar::handle:vertical {{
    background-color: {border};
    min-height: 36px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{ background-color: {muted}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QTabWidget::pane {{
    background-color: {panel};
    border: 1px solid {border};
    border-radius: 8px;
    top: -1px;
}}
QTabBar::tab {{
    background-color: {soft};
    color: {muted};
    border: 1px solid {border};
    padding: 7px 10px;
    min-width: 64px;
}}
QTabBar::tab:selected {{
    background-color: {panel};
    color: {PRIMARY_LIGHT if dark else PRIMARY};
    border-bottom-color: {panel};
    font-weight: 600;
}}
QToolTip {{
    background-color: {panel};
    color: {text};
    border: 1px solid {border};
    padding: 5px;
}}
"""


LIGHT_THEME: str = _build_theme(dark=False)
DARK_THEME: str = _build_theme(dark=True)


def apply_theme(app: QApplication, theme_name: str = "light") -> None:
    """Apply the light or dark QSS theme globally to *app*."""
    theme_name = "dark" if theme_name.lower() == "dark" else "light"
    app.setStyleSheet(DARK_THEME if theme_name == "dark" else LIGHT_THEME)
    app.setProperty("traffic_sign_theme", theme_name)


def toggle_theme(app: QApplication, current_theme: str) -> str:
    """Toggle the global theme and return the newly active theme name."""
    next_theme = "dark" if current_theme.lower() != "dark" else "light"
    apply_theme(app, next_theme)
    return next_theme
