import sys
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (QApplication, QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow,
                               QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from . import icons, theme
from .config import Config
from .pages.graphs import GraphsPage
from .pages.settings import SettingsPage
from .pages.training import TrainingPage
from .trainer import Trainer
from .widgets.common import label

TRAINING, GRAPHS, SETTINGS = range(3)


class AppState(QObject):
    runs_changed = Signal()     # a run was saved/renamed, or the output folder changed
    config_changed = Signal()

    def __init__(self):
        super().__init__()
        self.config = Config.load()
        Path(self.config.input_path).mkdir(parents=True, exist_ok=True)
        self.trainer = Trainer(self)


class StatusBox(QFrame):
    """Bottom of the sidebar: what the trainer is doing, visible from every page."""

    def __init__(self, trainer, on_click):
        super().__init__()
        self.setObjectName("StatusBox")
        self.setAttribute(Qt.WA_Hover)
        self.setCursor(Qt.PointingHandCursor)
        self._on_click = on_click
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(5)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.dot = QLabel()
        self.dot.setFixedSize(8, 8)
        self.title = label("", "FieldTitle")
        top.addWidget(self.dot)
        top.addWidget(self.title, 1)
        lay.addLayout(top)
        self.sub = label("", "Muted")
        lay.addWidget(self.sub)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        lay.addWidget(self.bar)

        trainer.started.connect(self._started)
        trainer.iteration.connect(self._progress)
        trainer.finished.connect(lambda run: self._set(theme.GOOD, "Run finished", run["name"]))
        trainer.stopped.connect(lambda: self._set(theme.MUTED, "Run stopped", "Nothing saved"))
        trainer.failed.connect(lambda _: self._set(theme.DANGER, "Failed", "Not saved"))
        self._set(theme.MUTED, "Idle", "No training running")

    def _set(self, color, title, sub, busy=False):
        self.dot.setStyleSheet(f"background: {color}; border-radius: 4px;")
        self.title.setText(title)
        self.sub.setText(sub)
        self.bar.setVisible(busy)

    def _started(self, run):
        self.bar.setRange(0, run["network_parameters"]["max_iterations"])
        self.bar.setValue(0)
        self._set(theme.ACCENT, "Training", "Starting…", busy=True)

    def _progress(self, iteration, metrics):
        self.bar.setValue(iteration)
        self.sub.setText(f"Iteration {iteration} · Categories: {metrics['num_categories']}")

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._on_click()


class Sidebar(QFrame):
    def __init__(self, state, pages):
        super().__init__()
        self.setObjectName("Sidebar")
        self.setFixedWidth(216)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 18, 12, 12)
        lay.setSpacing(2)

        brand = QVBoxLayout()
        brand.setContentsMargins(8, 0, 0, 0)
        brand.setSpacing(1)
        brand.addWidget(label("Dolphin Acoustics", "Brand"))
        brand.addWidget(label("Unpublished · v0.1", "BrandSub"))
        lay.addLayout(brand)
        lay.addSpacing(24)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.group.idClicked.connect(pages.setCurrentIndex)
        section = label("WORKSPACE", "Section")
        section.setContentsMargins(10, 0, 0, 6)
        lay.addWidget(section)
        self._nav(lay, "Training", "activity", TRAINING)
        self._nav(lay, "Graphs", "chart", GRAPHS)
        lay.addStretch()
        self._nav(lay, "Settings", "sliders", SETTINGS)
        lay.addSpacing(10)
        lay.addWidget(StatusBox(state.trainer, lambda: self.select(TRAINING)))
        self.select(TRAINING)
        self._pages = pages

    def _nav(self, layout, text, icon_name, index):
        btn = QPushButton("  " + text)
        btn.setObjectName("Nav")
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setIcon(icons.icon(icon_name, theme.MUTED, theme.TEXT))
        self.group.addButton(btn, index)
        layout.addWidget(btn)

    def select(self, index):
        self.group.button(index).setChecked(True)
        if hasattr(self, "_pages"):
            self._pages.setCurrentIndex(index)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dolphin Acoustics ARTwarp")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self.state = AppState()

        self.pages = QStackedWidget()
        self.training = TrainingPage(self.state)
        self.graphs = GraphsPage(self.state)
        self.settings = SettingsPage(self.state)
        for page in (self.training, self.graphs, self.settings):
            self.pages.addWidget(page)
        self.sidebar = Sidebar(self.state, self.pages)

        root = QWidget()
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.sidebar)
        lay.addWidget(self.pages, 1)
        self.setCentralWidget(root)

    def closeEvent(self, e):
        self.state.trainer.shutdown()
        super().closeEvent(e)


def _dark_palette():
    pal = QPalette()
    for role, color in ((QPalette.Window, theme.BG), (QPalette.Base, theme.BG), (QPalette.AlternateBase, theme.SURFACE),
                        (QPalette.Text, theme.TEXT), (QPalette.WindowText, theme.TEXT), (QPalette.Button, theme.RAISED),
                        (QPalette.ButtonText, theme.TEXT), (QPalette.Highlight, theme.ACCENT),
                        (QPalette.HighlightedText, "#ffffff"), (QPalette.PlaceholderText, theme.MUTED),
                        (QPalette.ToolTipBase, theme.RAISED), (QPalette.ToolTipText, theme.TEXT)):
        pal.setColor(role, QColor(color))
    return pal


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Dolphin Acoustics ARTwarp")
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setStyleSheet(theme.QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())