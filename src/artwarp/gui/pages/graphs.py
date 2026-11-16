"""
History of past runs. Displays list and shows charts stored in run.json for selected run.
Refreshes when run is saved/renamed. Reads run.json. Same widget as training.py for Header and results.

@author: Jaime Barquero Cebrian
"""

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QListWidget, QListWidgetItem, QStackedWidget, QVBoxLayout,
                               QWidget)

from artwarp.io import runs
from artwarp.gui.widgets.common import Card, PageHeader, RunHeader, label
from artwarp.gui.widgets.results import ResultsView

LIST_WIDTH = 270


class RunItem(QWidget):
    """Individual run in history list, built from run dictionary"""
    def __init__(self, run):
        super().__init__()
        self.setObjectName("RunItem")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)
        name = label("", "FieldTitle")
        name.setText(name.fontMetrics().elidedText(run["name"], Qt.ElideRight, LIST_WIDTH - 50))
        name.setToolTip(run["name"])
        created = datetime.fromisoformat(run["created"]).strftime("%d %b, %H:%M")
        meta = created
        if "summary" in run:
            meta += f"  ·  {run['summary']['training_time']:.1f} s"
            meta += f"  ·  {run['summary']['number_categories']} categories · {'converged' if run['summary']['converged'] else 'did not converge'}"
        lay.addWidget(name)
        lay.addWidget(label(meta, "Muted"))


class GraphsPage(QWidget):
    """Shows charts of selected run. List of rows and self._runs share order"""
    def __init__(self, state):
        super().__init__()
        self.state = state
        # List of saved runs
        self._runs = []
        self._current_id = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)
        root.addWidget(PageHeader("Graphs", "Results of saved runs. Works while a run is training."))

        body = QHBoxLayout()
        body.setSpacing(16)
        side = Card()
        side.setFixedWidth(LIST_WIDTH)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(8, 12, 8, 10)
        sl.setSpacing(8)
        head = QHBoxLayout()
        head.setContentsMargins(8, 0, 8, 0)
        head.addWidget(label("Saved runs", "FieldTitle"))
        head.addStretch()
        self.count = label("", "Muted")
        head.addWidget(self.count)
        sl.addLayout(head)
        self.list = QListWidget()
        self.list.setObjectName("RunList")
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.list.currentRowChanged.connect(self._select)
        sl.addWidget(self.list, 1)
        hint = label("Double-click a run to rename it.", "Muted")
        hint.setContentsMargins(8, 0, 8, 0)
        sl.addWidget(hint)
        body.addWidget(side)

        self.right = QStackedWidget()
        self.empty = label("", "Empty")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setWordWrap(True)
        self.right.addWidget(self.empty)
        view = QWidget()
        vl = QVBoxLayout(view)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(14)
        self.header = RunHeader()
        self.header.renamed.connect(self._rename)
        self.results = ResultsView()
        vl.addWidget(self.header)
        vl.addWidget(self.results, 1)
        self.right.addWidget(view)
        body.addWidget(self.right, 1)
        root.addLayout(body, 1)

        # Double click on list focuses on name, allowing renaming
        self.list.itemDoubleClicked.connect(lambda _: self.header.focus_name())
        state.runs_changed.connect(self.reload)
        self.reload()

    def reload(self):
        """Rebuild list from disk, selection by id if run exists else newest run"""
        self._runs = runs.list_runs(self.state.config.output_path)
        # Blocks signals to change state without firing new signals
        self.list.blockSignals(True)
        self.list.clear()
        for run in self._runs:
            item = QListWidgetItem()
            widget = RunItem(run)
            item.setSizeHint(widget.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, widget)
        self.list.blockSignals(False)
        self.count.setText(str(len(self._runs)))

        if not self._runs:
            self._current_id = None
            self.empty.setText(f"No saved runs in\n{self.state.config.output_path}\n\nStart one from Training.")
            self.right.setCurrentIndex(0)
            return
        ids = [r["id"] for r in self._runs]
        self.list.setCurrentRow(ids.index(self._current_id) if self._current_id in ids else 0)

    # TODO: "Use these parameters" in the run header -> switch to Training with the
    #   form filled from this run (pairs with the "Load from run" TODO in training.py).
    def _select(self, row):
        """Displays a selected run"""
        if row < 0:
            return
        run = self._runs[row]
        self.header.set_run(run)
        # keeps the grid / expanded state when only the name changed
        if run["id"] != self._current_id:
            self._current_id = run["id"]
            self.results.set_charts(run.get("charts", []))
        self.right.setCurrentIndex(1)

    def _rename(self, name):
        """Write new name for specific run """
        row = self.list.currentRow()
        if row >= 0:
            runs.rename(self._runs[row], name)
            self.state.runs_changed.emit()
