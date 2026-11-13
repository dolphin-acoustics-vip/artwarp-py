"""App-wide stylesheet."""
from string import Template

BG = "#14161b"
SIDEBAR = "#181a20"
SURFACE = "#1c1f26"
RAISED = "#252932"
HOVER = "#2b2f39"
BORDER = "#2a2e37"
BORDER_STRONG = "#3a3f4b"
TEXT = "#e6e7ea"
TEXT_2 = "#a3a8b2"
MUTED = "#6b717c"
ACCENT = "#3987e5"
ACCENT_HOVER = "#4c95ec"
DANGER = "#e66767"
GOOD = "#0ca30c"

# Categorical series colours: train (blue), validation (orange), other (aqua).
# Checked for colour-blind separation and >= 3:1 contrast on SURFACE.
SERIES = ["#3987e5", "#d95926", "#199e70"]

# Single-hue sequential ramp for heatmaps, low -> high.
SEQ_LOW = "#222a36"
SEQ_HIGH = "#78aef2"

_QSS = Template("""
QWidget { color: $TEXT; font-size: 13px; }
QMainWindow { background: $BG; }
QToolTip { background: $RAISED; color: $TEXT; border: 1px solid $BORDER_STRONG; padding: 4px 6px; }

QFrame#Sidebar { background: $SIDEBAR; border: none; border-right: 1px solid $BORDER; }
QLabel#Brand { font-size: 14px; font-weight: 600; }
QLabel#BrandSub { color: $MUTED; font-size: 11px; }
QLabel#Section { color: $MUTED; font-size: 10px; font-weight: 600; }
QPushButton#Nav {
    text-align: left; padding: 8px 10px; border: none; border-radius: 6px;
    background: transparent; color: $TEXT_2;
}
QPushButton#Nav:hover { background: $RAISED; color: $TEXT; }
QPushButton#Nav:checked { background: $RAISED; color: $TEXT; font-weight: 600; }
QFrame#StatusBox { background: $SURFACE; border: 1px solid $BORDER; border-radius: 8px; }
QFrame#StatusBox:hover { border-color: $BORDER_STRONG; }

QFrame#Card { background: $SURFACE; border: 1px solid $BORDER; border-radius: 8px; }
QFrame#ChartCard { background: $SURFACE; border: 1px solid $BORDER; border-radius: 8px; }
QFrame#ChartCard:hover { background: #1f222a; border-color: $BORDER_STRONG; }
QFrame#Divider { background: $BORDER; border: none; }
QLabel#CardTitle { font-size: 12px; font-weight: 600; }
QLabel#PageTitle { font-size: 22px; font-weight: 600; }
QLabel#ExpandedTitle { font-size: 16px; font-weight: 600; }
QLabel#Subtle { color: $TEXT_2; }
QLabel#Muted { color: $MUTED; font-size: 12px; }
QLabel#FieldTitle { font-weight: 600; }
QLabel#FieldLabel { color: $TEXT_2; font-size: 12px; }
QLabel#StatLabel { color: $MUTED; font-size: 11px; }
QLabel#StatValue { font-size: 20px; font-weight: 600; }
QLabel#Error { color: $DANGER; font-size: 12px; }
QLabel#Empty { color: $MUTED; font-size: 13px; }

QLineEdit {
    background: $BG; border: 1px solid $BORDER; border-radius: 6px;
    padding: 7px 10px; selection-background-color: $ACCENT;
}
QLineEdit:hover { border-color: $BORDER_STRONG; }
QLineEdit:focus { border-color: $ACCENT; }
QLineEdit:disabled { color: $MUTED; }
QAbstractSpinBox:disabled { color: $MUTED; }
QCheckBox:disabled { color: $MUTED; }
QLabel#FieldLabel:disabled { color: $MUTED; }
QLineEdit#TitleEdit {
    background: transparent; border: 1px solid transparent;
    font-size: 18px; font-weight: 600; padding: 2px 6px;
}
QLineEdit#TitleEdit:hover { border-color: $BORDER; }
QLineEdit#TitleEdit:focus { border-color: $ACCENT; background: $BG; }

QPushButton {
    background: $RAISED; border: 1px solid $BORDER; border-radius: 6px;
    padding: 7px 14px; color: $TEXT;
}
QPushButton:hover { background: $HOVER; border-color: $BORDER_STRONG; }
QPushButton:pressed { background: $RAISED; }
QPushButton:disabled { color: $MUTED; background: $SURFACE; }
QPushButton#Primary { background: $ACCENT; border-color: $ACCENT; color: #ffffff; font-weight: 600; }
QPushButton#Primary:hover { background: $ACCENT_HOVER; border-color: $ACCENT_HOVER; }
QPushButton#Primary:disabled { background: #24364d; border-color: #24364d; color: #7d93ad; }
QPushButton#Stop { color: $DANGER; }
QPushButton#Ghost { background: transparent; border: 1px solid transparent; color: $TEXT_2; padding: 6px 10px; }
QPushButton#Ghost:hover { background: $RAISED; color: $TEXT; }

QProgressBar { background: $RAISED; border: none; border-radius: 2px; min-height: 4px; max-height: 4px; }
QProgressBar::chunk { background: $ACCENT; border-radius: 2px; }

QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: $BORDER_STRONG; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

QListWidget#RunList { background: transparent; border: none; outline: 0; }
QListWidget#RunList::item { border-radius: 6px; margin: 1px 0; }
QListWidget#RunList::item:hover { background: #21252d; }
QListWidget#RunList::item:selected { background: $RAISED; }
QWidget#RunItem, QWidget#RunItem QLabel { background: transparent; }
""")

QSS = _QSS.substitute({k: v for k, v in globals().items() if k.isupper() and isinstance(v, str)})
