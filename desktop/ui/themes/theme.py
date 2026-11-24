"""Modern Light & Dark Qt 6 stylesheets for Nuranova Mocap Studio."""

from __future__ import annotations

LIGHT_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #F4F6F9;
    color: #1A232E;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QFrame#Sidebar {
    background-color: #FFFFFF;
    border-right: 1px solid #DCE3EC;
}
QFrame#Card, QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #DCE3EC;
    border-radius: 8px;
}
QGroupBox {
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #0D7C7C;
}
QListWidget#NavList {
    background-color: transparent;
    border: none;
    outline: none;
}
QListWidget#NavList::item {
    padding: 7px 12px;
    border-radius: 6px;
    margin: 1px 6px;
    color: #2E3B4A;
}
QListWidget#NavList::item:hover {
    background-color: #EBF2F7;
}
QListWidget#NavList::item:selected {
    background-color: #0D7C7C;
    color: #FFFFFF;
    font-weight: 600;
}
QPushButton {
    background-color: #FFFFFF;
    color: #1A232E;
    border: 1px solid #C8D3E0;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #EDF3F8;
    border-color: #0D7C7C;
}
QPushButton:pressed {
    background-color: #DCE8F0;
}
QPushButton#PrimaryButton {
    background-color: #0D7C7C;
    color: #FFFFFF;
    border: 1px solid #0B6A6A;
    font-weight: 600;
}
QPushButton#PrimaryButton:hover {
    background-color: #0F8F8F;
}
QPushButton#DangerButton {
    background-color: #D32F2F;
    color: #FFFFFF;
    border: 1px solid #B71C1C;
    font-weight: 600;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background-color: #FFFFFF;
    color: #1A232E;
    border: 1px solid #C8D3E0;
    border-radius: 5px;
    padding: 5px 8px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #0D7C7C;
}
QTableWidget, QTreeWidget {
    background-color: #FFFFFF;
    alternate-background-color: #F8FAFC;
    border: 1px solid #DCE3EC;
    border-radius: 6px;
    gridline-color: #E6ECF2;
}
QHeaderView::section {
    background-color: #EEF2F7;
    color: #2E3B4A;
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid #DCE3EC;
    font-weight: 600;
}
QTabWidget::pane {
    border: 1px solid #DCE3EC;
    background-color: #FFFFFF;
    border-radius: 6px;
}
QTabBar::tab {
    background-color: #E9EFF5;
    color: #2E3B4A;
    padding: 7px 14px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #0D7C7C;
    font-weight: 600;
}
QStatusBar {
    background-color: #FFFFFF;
    border-top: 1px solid #DCE3EC;
    color: #4A5A6B;
}
QProgressBar {
    border: 1px solid #C8D3E0;
    border-radius: 4px;
    text-align: center;
    background-color: #EEF2F7;
}
QProgressBar::chunk {
    background-color: #0D7C7C;
    border-radius: 3px;
}
QLabel#PageTitle {
    font-size: 20px;
    font-weight: 700;
    color: #12202E;
}
QLabel#Subtitle {
    color: #5A6B7D;
    font-size: 12px;
}
"""

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #11161D;
    color: #E6EDF5;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QFrame#Sidebar {
    background-color: #171E27;
    border-right: 1px solid #263241;
}
QFrame#Card, QGroupBox {
    background-color: #1A222D;
    border: 1px solid #2A3646;
    border-radius: 8px;
}
QGroupBox {
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #26B5B5;
}
QListWidget#NavList {
    background-color: transparent;
    border: none;
    outline: none;
}
QListWidget#NavList::item {
    padding: 7px 12px;
    border-radius: 6px;
    margin: 1px 6px;
    color: #C3D0DF;
}
QListWidget#NavList::item:hover {
    background-color: #222D3B;
}
QListWidget#NavList::item:selected {
    background-color: #158A8A;
    color: #FFFFFF;
    font-weight: 600;
}
QPushButton {
    background-color: #212B38;
    color: #E6EDF5;
    border: 1px solid #334254;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #2A3646;
    border-color: #26B5B5;
}
QPushButton#PrimaryButton {
    background-color: #158A8A;
    color: #FFFFFF;
    border: 1px solid #1BA3A3;
    font-weight: 600;
}
QPushButton#PrimaryButton:hover {
    background-color: #1BA3A3;
}
QPushButton#DangerButton {
    background-color: #C62828;
    color: #FFFFFF;
    border: 1px solid #E53935;
    font-weight: 600;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background-color: #141B23;
    color: #E6EDF5;
    border: 1px solid #324153;
    border-radius: 5px;
    padding: 5px 8px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #26B5B5;
}
QTableWidget, QTreeWidget {
    background-color: #161D26;
    alternate-background-color: #1B2430;
    border: 1px solid #2A3646;
    border-radius: 6px;
    gridline-color: #253140;
}
QHeaderView::section {
    background-color: #1F2834;
    color: #D2DCE8;
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid #2A3646;
    font-weight: 600;
}
QTabWidget::pane {
    border: 1px solid #2A3646;
    background-color: #1A222D;
    border-radius: 6px;
}
QTabBar::tab {
    background-color: #171E27;
    color: #B4C2D2;
    padding: 7px 14px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #1A222D;
    color: #26B5B5;
    font-weight: 600;
}
QStatusBar {
    background-color: #171E27;
    border-top: 1px solid #263241;
    color: #9FB0C4;
}
QProgressBar {
    border: 1px solid #324153;
    border-radius: 4px;
    text-align: center;
    background-color: #141B23;
}
QProgressBar::chunk {
    background-color: #1CA3A3;
    border-radius: 3px;
}
QLabel#PageTitle {
    font-size: 20px;
    font-weight: 700;
    color: #F0F5FA;
}
QLabel#Subtitle {
    color: #8EA0B5;
    font-size: 12px;
}
"""


def get_stylesheet(theme: str = "light") -> str:
    return DARK_STYLESHEET if theme == "dark" else LIGHT_STYLESHEET
