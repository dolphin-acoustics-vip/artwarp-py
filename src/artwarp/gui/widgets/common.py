"""Helpers shared by the pages."""
from datetime import datetime

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from .. import icons, theme


def label(text="", name=None, wrap=False):
    lbl = QLabel(text)
    if name:
        lbl.setObjectName(name)
    lbl.setWordWrap(wrap)
    return lbl


def set_role(widget, name):
    """Change objectName and re-apply the stylesheet (e.g. Muted -> Error)."""
    widget.setObjectName(name)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def open_folder(path):
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")


class PageHeader(QWidget):
    def __init__(self, title, subtitle=""):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(label(title, "PageTitle"))
        if subtitle:
            lay.addWidget(label(subtitle, "Subtle"))


class StatBlock(Card):
    def __init__(self, title):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        lay.addWidget(label(title.upper(), "StatLabel"))
        self.value = label("–", "StatValue")
        lay.addWidget(self.value)

    def set_value(self, text):
        self.value.setText(text)


class RunHeader(QWidget):
    """Editable run name + one line of metadata. Emits renamed(new_name)."""

    renamed = Signal(str)

    def __init__(self):
        super().__init__()
        self._name = ""
        self._path = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        text = QVBoxLayout()
        text.setSpacing(2)
        name_row = QHBoxLayout()
        name_row.setSpacing(4)
        self.name = QLineEdit()
        self.name.setObjectName("TitleEdit")
        self.name.setToolTip("Click to rename")
        self.name.editingFinished.connect(self._commit)
        self.name.textChanged.connect(self._fit)
        pencil = QLabel()
        pencil.setPixmap(icons.pixmap("pencil", theme.MUTED, 13))
        name_row.addWidget(self.name)
        name_row.addWidget(pencil)
        name_row.addStretch()
        self.meta = label("", "Muted")
        self.meta.setContentsMargins(7, 0, 0, 0)
        text.addLayout(name_row)
        text.addWidget(self.meta)
        lay.addLayout(text, 1)

        folder = QPushButton(" Open folder")
        folder.setObjectName("Ghost")
        folder.setIcon(icons.icon("folder", theme.TEXT_2))
        folder.clicked.connect(lambda: self._path and open_folder(self._path))
        lay.addWidget(folder, 0, Qt.AlignTop)

    def set_run(self, run):
        self._name = run["name"]
        self._path = run.get("path")
        self.name.setText(self._name)
        created = datetime.fromisoformat(run["created"]).strftime("%d %b %Y, %H:%M")
        parts = [f"Ran {created}", f"{run['summary']['training_time']:.1f} s", f"{run['summary']['number_iterations']} iterations"]
        if "summary" in run:
            parts.append(f"{run['summary']['number_categories']} Categories · {'Training Converged' if run['summary']['converged'] else 'Training did not converge'}")
        self.meta.setText("  ·  ".join(parts))

    def set_name(self, name):
        self._name = name
        self.name.setText(name)

    def focus_name(self):
        self.name.setFocus()
        self.name.selectAll()

    def _commit(self):
        new = self.name.text().strip()
        if not new:
            self.name.setText(self._name)
        elif new != self._name:
            self._name = new
            self.renamed.emit(new)
        self.name.clearFocus()

    def _fit(self, text):
        self.name.ensurePolished()
        width = self.name.fontMetrics().horizontalAdvance(text or " ") + 24
        self.name.setFixedWidth(max(120, min(width, 560)))
