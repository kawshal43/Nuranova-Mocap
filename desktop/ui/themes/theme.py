"""Modern Pro-Studio Dark & Light Qt 6 stylesheets for Nuranova Mocap Studio."""

from __future__ import annotations

LIGHT_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #F1F5F9;
    color: #0F172A;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QFrame#Sidebar {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}
QFrame#WorkflowBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
}
QFrame#BannerCard {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #E6F8F6, stop:0.65 #FFFFFF, stop:1 #EFF6FF);
    border: 1px solid #B7E4DF;
    border-radius: 14px;
}
QFrame#ActionCard, QFrame#ProjectCard, QFrame#MetricCard, QFrame#Card, QFrame#HeroCard, QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #DCE4EE;
    border-radius: 12px;
}
QFrame#ActionCard:hover, QFrame#ProjectCard:hover {
    border: 1px solid #0D9488;
    background-color: #F8FFFE;
}
QFrame#QrCard {
    background-color: #FFFFFF;
    border: 2px solid #0D9488;
    border-radius: 12px;
}
QLabel#IconBadge {
    background-color: #E6F7F5;
    color: #0F766E;
    border: 1px solid #B6E6E0;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 700;
}
QLabel#MetricValue {
    font-size: 17px;
    font-weight: 800;
    color: #0F172A;
}
QLabel#MetricTitle {
    font-size: 11px;
    font-weight: 700;
    color: #64748B;
    letter-spacing: 0.4px;
}
QGroupBox {
    margin-top: 14px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #0F766E;
}
QPushButton#NavPrimary {
    background-color: transparent;
    color: #334155;
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 13px;
}
QPushButton#NavPrimary:hover {
    background-color: #EFF6F8;
    color: #0F172A;
}
QPushButton#NavPrimary[active="true"] {
    background-color: #0D9488;
    color: #FFFFFF;
}
QPushButton#SectionToggle {
    background-color: transparent;
    color: #64748B;
    border: none;
    padding: 6px 8px;
    text-align: left;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.6px;
}
QPushButton#SectionToggle:hover {
    color: #0D9488;
}
QPushButton#StepPill {
    background-color: #F1F5F9;
    color: #334155;
    border: 1px solid #CBD5E1;
    border-radius: 14px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton#StepPill:hover {
    border-color: #0D9488;
}
QPushButton#StepPill[stepState="done"] {
    background-color: #DCFCE7;
    color: #166534;
    border-color: #86EFAC;
}
QPushButton#StepPill[stepState="active"] {
    background-color: #0D9488;
    color: #FFFFFF;
    border-color: #0F766E;
}
QPushButton#StepPill[stepState="recording"] {
    background-color: #DC2626;
    color: #FFFFFF;
    border-color: #B91C1C;
}
QListWidget#NavList {
    background-color: transparent;
    border: none;
    outline: none;
}
QListWidget#NavList::item {
    padding: 6px 10px;
    border-radius: 7px;
    margin: 1px 4px;
    color: #334155;
    font-size: 12px;
}
QListWidget#NavList::item:hover {
    background-color: #EFF6F8;
}
QListWidget#NavList::item:selected {
    background-color: #0D9488;
    color: #FFFFFF;
    font-weight: 600;
}
QPushButton {
    background-color: #FFFFFF;
    color: #0F172A;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #F1F5F9;
    border-color: #0D9488;
}
QPushButton:pressed {
    background-color: #E2E8F0;
}
QPushButton#PrimaryButton {
    background-color: #0D9488;
    color: #FFFFFF;
    border: 1px solid #0F766E;
    font-weight: 700;
}
QPushButton#PrimaryButton:hover {
    background-color: #14B8A6;
}
QPushButton#DangerButton {
    background-color: #DC2626;
    color: #FFFFFF;
    border: 1px solid #B91C1C;
    font-weight: 700;
}
QPushButton#DangerButton:hover {
    background-color: #EF4444;
}
QLabel#BadgeOk {
    background-color: #DCFCE7;
    color: #166534;
    border: 1px solid #86EFAC;
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 700;
    font-size: 11px;
}
QLabel#BadgeWarn {
    background-color: #FEF3C7;
    color: #92400E;
    border: 1px solid #FCD34D;
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 700;
    font-size: 11px;
}
QLabel#BadgeErr {
    background-color: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FCA5A5;
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 700;
    font-size: 11px;
}
QLabel#PinDisplay {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 6px;
    color: #0F766E;
    background-color: #F0FDFA;
    border: 1px solid #99F6E4;
    border-radius: 10px;
    padding: 6px 14px;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background-color: #FFFFFF;
    color: #0F172A;
    border: 1px solid #CBD5E1;
    border-radius: 7px;
    padding: 5px 9px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #0D9488;
}
QTableWidget, QTreeWidget {
    background-color: #FFFFFF;
    alternate-background-color: #F8FAFC;
    border: 1px solid #DCE4EE;
    border-radius: 10px;
    gridline-color: #E2E8F0;
}
QHeaderView::section {
    background-color: #F1F5F9;
    color: #334155;
    padding: 7px 8px;
    border: none;
    border-bottom: 1px solid #DCE4EE;
    font-weight: 700;
}
QTabWidget::pane {
    border: 1px solid #DCE4EE;
    background-color: #FFFFFF;
    border-radius: 10px;
}
QTabBar::tab {
    background-color: #E2E8F0;
    color: #334155;
    padding: 8px 16px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 3px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #0F766E;
}
QStatusBar {
    background-color: #FFFFFF;
    border-top: 1px solid #E2E8F0;
    color: #475569;
}
QScrollArea {
    border: none;
    background: transparent;
}
QProgressBar {
    border: 1px solid #CBD5E1;
    border-radius: 5px;
    text-align: center;
    background-color: #E2E8F0;
}
QProgressBar::chunk {
    background-color: #0D9488;
    border-radius: 4px;
}
QLabel#PageTitle {
    font-size: 22px;
    font-weight: 800;
    color: #0F172A;
}
QLabel#Subtitle {
    color: #64748B;
    font-size: 12px;
}
"""

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0A0F16;
    color: #E6EDF5;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QFrame#Sidebar {
    background-color: #0F1620;
    border-right: 1px solid #1E2A3A;
}
QFrame#WorkflowBar {
    background-color: #0F1620;
    border-bottom: 1px solid #1E2A3A;
}
QFrame#BannerCard {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0E2B33, stop:0.55 #131D2A, stop:1 #162033);
    border: 1px solid #1F4E5A;
    border-radius: 14px;
}
QFrame#ActionCard, QFrame#ProjectCard, QFrame#MetricCard, QFrame#Card, QFrame#HeroCard, QGroupBox {
    background-color: #121B26;
    border: 1px solid #223044;
    border-radius: 12px;
}
QFrame#ActionCard:hover, QFrame#ProjectCard:hover {
    border: 1px solid #2DD4BF;
    background-color: #162231;
}
QFrame#QrCard {
    background-color: #FFFFFF;
    border: 2px solid #2DD4BF;
    border-radius: 12px;
}
QLabel#IconBadge {
    background-color: #102F35;
    color: #2DD4BF;
    border: 1px solid #1E5A63;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 700;
}
QLabel#MetricValue {
    font-size: 17px;
    font-weight: 800;
    color: #F8FAFC;
}
QLabel#MetricTitle {
    font-size: 11px;
    font-weight: 700;
    color: #7E92A9;
    letter-spacing: 0.4px;
}
QGroupBox {
    margin-top: 14px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #2DD4BF;
}
QPushButton#NavPrimary {
    background-color: transparent;
    color: #B8C7D9;
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 13px;
}
QPushButton#NavPrimary:hover {
    background-color: #182332;
    color: #FFFFFF;
}
QPushButton#NavPrimary[active="true"] {
    background-color: #0F8B8D;
    color: #FFFFFF;
    border-color: #2DD4BF;
}
QPushButton#SectionToggle {
    background-color: transparent;
    color: #788CA3;
    border: none;
    padding: 6px 8px;
    text-align: left;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.6px;
}
QPushButton#SectionToggle:hover {
    color: #2DD4BF;
}
QPushButton#StepPill {
    background-color: #16202C;
    color: #A2B5CA;
    border: 1px solid #263649;
    border-radius: 14px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton#StepPill:hover {
    border-color: #2DD4BF;
    color: #FFFFFF;
}
QPushButton#StepPill[stepState="done"] {
    background-color: #0F2E22;
    color: #4ADE80;
    border-color: #1E6146;
}
QPushButton#StepPill[stepState="active"] {
    background-color: #0F8B8D;
    color: #FFFFFF;
    border-color: #2DD4BF;
}
QPushButton#StepPill[stepState="recording"] {
    background-color: #DC2626;
    color: #FFFFFF;
    border-color: #F87171;
}
QListWidget#NavList {
    background-color: transparent;
    border: none;
    outline: none;
}
QListWidget#NavList::item {
    padding: 6px 10px;
    border-radius: 7px;
    margin: 1px 4px;
    color: #B4C3D5;
    font-size: 12px;
}
QListWidget#NavList::item:hover {
    background-color: #182332;
}
QListWidget#NavList::item:selected {
    background-color: #0F8B8D;
    color: #FFFFFF;
    font-weight: 600;
}
QPushButton {
    background-color: #192432;
    color: #E6EDF5;
    border: 1px solid #2B3B50;
    border-radius: 8px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #223042;
    border-color: #2DD4BF;
}
QPushButton#PrimaryButton {
    background-color: #0D9488;
    color: #FFFFFF;
    border: 1px solid #2DD4BF;
    font-weight: 700;
}
QPushButton#PrimaryButton:hover {
    background-color: #14B8A6;
}
QPushButton#DangerButton {
    background-color: #DC2626;
    color: #FFFFFF;
    border: 1px solid #F87171;
    font-weight: 700;
}
QPushButton#DangerButton:hover {
    background-color: #EF4444;
}
QLabel#BadgeOk {
    background-color: #0F2E22;
    color: #4ADE80;
    border: 1px solid #1E6146;
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 700;
    font-size: 11px;
}
QLabel#BadgeWarn {
    background-color: #362510;
    color: #FBBF24;
    border: 1px solid #78501A;
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 700;
    font-size: 11px;
}
QLabel#BadgeErr {
    background-color: #3B1519;
    color: #F87171;
    border: 1px solid #7F262D;
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 700;
    font-size: 11px;
}
QLabel#PinDisplay {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 6px;
    color: #2DD4BF;
    background-color: #0C1520;
    border: 1px solid #1D555C;
    border-radius: 10px;
    padding: 6px 14px;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background-color: #0E151F;
    color: #E6EDF5;
    border: 1px solid #28384D;
    border-radius: 7px;
    padding: 5px 9px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #2DD4BF;
}
QTableWidget, QTreeWidget {
    background-color: #101822;
    alternate-background-color: #151F2B;
    border: 1px solid #223044;
    border-radius: 10px;
    gridline-color: #1D2A3A;
}
QHeaderView::section {
    background-color: #172230;
    color: #D2DCE8;
    padding: 7px 8px;
    border: none;
    border-bottom: 1px solid #223044;
    font-weight: 700;
}
QTabWidget::pane {
    border: 1px solid #223044;
    background-color: #121B26;
    border-radius: 10px;
}
QTabBar::tab {
    background-color: #0F1620;
    color: #93A6BC;
    padding: 8px 16px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 3px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #121B26;
    color: #2DD4BF;
}
QStatusBar {
    background-color: #0F1620;
    border-top: 1px solid #1E2A3A;
    color: #8DA0B5;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: transparent;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: #2B3B50;
    border-radius: 4px;
    min-height: 24px;
}
QProgressBar {
    border: 1px solid #28384D;
    border-radius: 5px;
    text-align: center;
    background-color: #0E151F;
}
QProgressBar::chunk {
    background-color: #14B8A6;
    border-radius: 4px;
}
QLabel#PageTitle {
    font-size: 22px;
    font-weight: 800;
    color: #F8FAFC;
}
QLabel#Subtitle {
    color: #8CA0B6;
    font-size: 12px;
}
"""


def get_stylesheet(theme: str = "dark") -> str:
    """Keep shared controls and studio chrome consistent across both palettes."""
    dark = theme == "dark"
    colors = {
        "TEXT": "#E6EDF5" if dark else "#14283E",
        "MUTED": "#91A5BD" if dark else "#5C7087",
        "SURFACE": "#142131" if dark else "#FFFFFF",
        "BORDER": "#2B3D52" if dark else "#CFDAE6",
        "ACCENT": "#5EEAD4" if dark else "#0F766E",
        "ACTIVE": "#153A42" if dark else "#DDF4F0",
        "DISABLED": "#597088" if dark else "#8999AB",
        "HOVER": "#1B3044" if dark else "#EDF7F6",
    }
    polish = """
        QLabel { background: transparent; }
        QWidget#SidebarSpacer { background: transparent; }
        QLabel#BrandName { font-size: 17px; font-weight: 800; letter-spacing: 1px; color: $TEXT; }
        QLabel#BrandEdition { font-size: 8px; font-weight: 600; color: $MUTED; padding-top: 5px; }
        QLabel#WorkspaceTitle { font-size: 13px; font-weight: 700; color: $TEXT; }
        QLabel#WorkspaceContext { font-size: 10px; color: $MUTED; }
        QLabel#HeroEyebrow { font-size: 10px; font-weight: 700; color: $ACCENT; letter-spacing: 1.5px; }
        QLabel#HeroTitle { font-size: 28px; font-weight: 700; color: $TEXT; }
        QFrame#WorkflowBar { min-height: 44px; }
        QPushButton#NavPrimary { padding: 11px 14px; font-size: 12px; }
        QPushButton#NavPrimary[active="true"] { background: $ACTIVE; border: 1px solid $BORDER; color: $ACCENT; }
        QPushButton#NavPrimary:focus { border-color: $ACCENT; }
        QPushButton#SectionToggle { padding-top: 10px; padding-bottom: 10px; }
        QPushButton:focus, QToolButton:focus { border: 1px solid $ACCENT; }
        QPushButton:disabled, QToolButton:disabled { background: $SURFACE; color: $DISABLED; border-color: $BORDER; }
        QPushButton#DangerButton:disabled { background: $SURFACE; color: $DISABLED; border-color: $BORDER; }
        QPushButton#PrimaryButton:pressed { background: #115E59; }
        QPushButton#PrimaryButton:disabled { background: $SURFACE; color: $DISABLED; border-color: $BORDER; }
        QToolButton { background: $SURFACE; color: $TEXT; border: 1px solid $BORDER; border-radius: 6px; padding: 5px; }
        QToolButton:hover { background: $HOVER; border-color: $ACCENT; }
        QToolButton:checked { background: $ACTIVE; color: $ACCENT; border-color: $ACCENT; }
        QTabBar::tab { padding: 9px 16px; border-bottom: 2px solid transparent; }
        QTabBar::tab:selected { border-bottom: 2px solid $ACCENT; }
        QTabBar::tab:hover { color: $ACCENT; }
        QCheckBox { spacing: 7px; background: transparent; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid $BORDER; border-radius: 4px; background: $SURFACE; }
        QCheckBox::indicator:checked { background: #0D9488; border: 2px solid $ACCENT; }
        QCheckBox::indicator:hover { border-color: $ACCENT; }
        QCheckBox::indicator:disabled { background: $SURFACE; border-color: $DISABLED; }
        QSlider::groove:horizontal { height: 5px; background: $BORDER; border-radius: 2px; }
        QSlider::sub-page:horizontal { background: #0D9488; border-radius: 2px; }
        QSlider::handle:horizontal { width: 13px; margin: -5px 0; background: $ACCENT; border: 2px solid $SURFACE; border-radius: 7px; }
        QSlider::handle:horizontal:hover { border-color: $ACCENT; }
        QScrollBar::add-line, QScrollBar::sub-line { height: 0px; width: 0px; }
        QToolTip { background: $SURFACE; color: $TEXT; border: 1px solid $BORDER; padding: 7px; }
        QStatusBar { padding: 3px 8px; font-size: 11px; }
    """
    for token, value in colors.items():
        polish = polish.replace("$" + token, value)
    return (DARK_STYLESHEET if dark else LIGHT_STYLESHEET) + polish
