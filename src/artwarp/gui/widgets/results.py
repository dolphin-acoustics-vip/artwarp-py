"""4x5 chart grid and expanded chart, used by Training and Graphs."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QStackedWidget, QVBoxLayout, QWidget)

from .. import icons, theme
from .chart import ChartWidget
from .common import Card, label

ROWS, COLUMNS = 4, 5


class ChartCard(QFrame):
    clicked = Signal(int)

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.setObjectName("ChartCard")
        self.setAttribute(Qt.WA_Hover)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(140)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 10, 8)
        lay.setSpacing(4)

        top = QHBoxLayout()
        self.title = label("", "CardTitle")
        self.expand = QLabel()
        self.expand.setFixedSize(14, 14)
        self._expand_icon = icons.pixmap("expand", theme.MUTED, 13)
        top.addWidget(self.title, 1)
        top.addWidget(self.expand)
        lay.addLayout(top)

        self.chart = ChartWidget(compact=True)
        lay.addWidget(self.chart, 1)

    def set_spec(self, spec):
        self.title.setText(spec["title"])
        self.chart.set_spec(spec)

    def enterEvent(self, e):
        self.expand.setPixmap(self._expand_icon)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.expand.setPixmap(QPixmap())
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit(self.index)


class ExpandedView(QWidget):
    closed = Signal()

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.charts, self.index = [], 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        back = self._button(" All charts", "arrow-left")
        back.clicked.connect(self.closed.emit)
        bar.addWidget(back, 0, Qt.AlignTop)
        bar.addSpacing(8)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title = label("", "ExpandedTitle")
        self.desc = label("", "Muted")
        titles.addWidget(self.title)
        titles.addWidget(self.desc)
        bar.addLayout(titles, 1)
        self.hint = label("Esc to close  ·  ← → to switch", "Muted")
        bar.addWidget(self.hint, 0, Qt.AlignVCenter)
        bar.addSpacing(12)
        prev = self._button("", "chevron-left")
        prev.clicked.connect(lambda: self._go(self.index - 1))
        self.count = label("", "Subtle")
        self.count.setAlignment(Qt.AlignCenter)
        self.count.setMinimumWidth(self.count.fontMetrics().horizontalAdvance("00 / 00") + 8)
        nxt = self._button("", "chevron-right")
        nxt.clicked.connect(lambda: self._go(self.index + 1))
        for w in (prev, self.count, nxt):
            bar.addWidget(w, 0, Qt.AlignVCenter)
        lay.addLayout(bar)

        card = Card()
        inner = QVBoxLayout(card)
        inner.setContentsMargins(16, 16, 16, 12)
        self.chart = ChartWidget(compact=False)
        inner.addWidget(self.chart)
        lay.addWidget(card, 1)

    @staticmethod
    def _button(text, icon_name):
        btn = QPushButton(text)
        btn.setObjectName("Ghost")
        btn.setIcon(icons.icon(icon_name, theme.TEXT_2))
        btn.setFocusPolicy(Qt.NoFocus)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def show_chart(self, charts, index):
        self.charts = charts
        self._go(index)

    def _go(self, index):
        if not self.charts:
            return
        self.index = index % len(self.charts)
        spec = self.charts[self.index]
        self.title.setText(spec["title"])
        self.desc.setText(spec.get("desc", ""))
        self.count.setText(f"{self.index + 1} / {len(self.charts)}")
        self.chart.set_spec(spec)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.closed.emit()
        elif e.key() == Qt.Key_Left:
            self._go(self.index - 1)
        elif e.key() == Qt.Key_Right:
            self._go(self.index + 1)
        else:
            super().keyPressEvent(e)


class ResultsView(QStackedWidget):
    """Page 0: scrollable grid of chart cards. Page 1: one chart, expanded."""

    def __init__(self):
        super().__init__()
        self.charts = []

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 4, 0)
        grid.setSpacing(12)
        self.cards = []
        for i in range(ROWS * COLUMNS):
            card = ChartCard(i)
            card.clicked.connect(self._open)
            grid.addWidget(card, i // COLUMNS, i % COLUMNS)
            self.cards.append(card)
        for c in range(COLUMNS):
            grid.setColumnStretch(c, 1)
        for r in range(ROWS):
            grid.setRowStretch(r, 1)
        scroll.setWidget(holder)

        self.expanded = ExpandedView()
        self.expanded.closed.connect(lambda: self.setCurrentIndex(0))
        self.addWidget(scroll)
        self.addWidget(self.expanded)

    def set_charts(self, charts):
        self.charts = charts
        for i, card in enumerate(self.cards):
            card.setVisible(i < len(charts))
            if i < len(charts):
                card.set_spec(charts[i])
        self.setCurrentIndex(0)

    def _open(self, index):
        self.expanded.show_chart(self.charts, index)
        self.setCurrentIndex(1)
        self.expanded.setFocus()
