from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (QHBoxLayout, QLineEdit, QProgressBar, QPushButton, QStackedWidget,
                               QVBoxLayout, QWidget)

from .. import icons, runs
from ..widgets.chart import ChartWidget
from ..widgets.common import Card, PageHeader, RunHeader, StatBlock, label
from ..widgets.results import ResultsView

MAX_SECONDS = 3600
EMPTY, LIVE, RESULTS = range(3)


def _path_label(title, parent):
    col = QVBoxLayout()
    col.setSpacing(4)
    col.addWidget(label(title, "FieldLabel"))
    value = label("", "Subtle")
    col.addWidget(value)
    parent.addLayout(col, 1)
    return value


class TrainingPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.trainer = state.trainer
        self._run = None
        self._live = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)
        root.addWidget(PageHeader("Training", "All parameters are editable. Currently they show default values."))
        root.addWidget(self._build_controls())

        self.body = QStackedWidget()
        self.empty = label("No run yet. Enter a duration and press Start; results show up here and in Graphs.", "Empty")
        self.empty.setAlignment(Qt.AlignCenter)
        self.body.addWidget(self.empty)
        self.body.addWidget(self._build_live())
        results = QWidget()
        rl = QVBoxLayout(results)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(14)
        self.header = RunHeader()
        self.header.renamed.connect(self._rename)
        self.results = ResultsView()
        rl.addWidget(self.header)
        rl.addWidget(self.results, 1)
        self.body.addWidget(results)
        root.addWidget(self.body, 1)

        self.trainer.started.connect(self._on_started)
        self.trainer.progress.connect(self._on_progress)
        self.trainer.iteration.connect(self._on_iteration)
        self.trainer.finished.connect(self._on_finished)
        self.trainer.stopped.connect(lambda: self._on_idle("Run stopped. Nothing was saved."))
        self.trainer.failed.connect(self._on_failed)
        state.config_changed.connect(self._show_paths)
        state.runs_changed.connect(self._refresh_name)
        self._show_paths()

    def _build_controls(self):
        card = Card()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(24)
        dur = QVBoxLayout()
        dur.setSpacing(4)
        dur.addWidget(label("Duration (seconds)", "FieldLabel"))
        self.duration = QLineEdit()
        self.duration.setPlaceholderText("e.g. 20")
        self.duration.setValidator(QIntValidator(1, MAX_SECONDS, self))
        self.duration.setFixedWidth(150)
        self.duration.returnPressed.connect(self._start)
        dur.addWidget(self.duration)
        row.addLayout(dur)
        self.input_lbl = _path_label("Input data", row)
        self.output_lbl = _path_label("Saves to", row)

        self.start_btn = QPushButton(" Start training")
        self.start_btn.setObjectName("Primary")
        self.start_btn.setIcon(icons.icon("play", "#ffffff", size=14))
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton(" Stop")
        self.stop_btn.setObjectName("Stop")
        self.stop_btn.setIcon(icons.icon("square", "#e66767", size=14))
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.clicked.connect(self.trainer.stop)
        self.stop_btn.hide()
        row.addWidget(self.start_btn, 0, Qt.AlignBottom)
        row.addWidget(self.stop_btn, 0, Qt.AlignBottom)
        lay.addLayout(row)

        self.error = label("", "Error")
        self.error.hide()
        lay.addWidget(self.error)

        self.progress_box = QWidget()
        pl = QVBoxLayout(self.progress_box)
        pl.setContentsMargins(0, 4, 0, 0)
        pl.setSpacing(6)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress_lbl = label("", "Muted")
        pl.addWidget(self.progress)
        pl.addWidget(self.progress_lbl)
        self.progress_box.hide()
        lay.addWidget(self.progress_box)
        return card

    def _build_live(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        stats = QHBoxLayout()
        stats.setSpacing(12)
        self.stats = {key: StatBlock(title) for key, title in
                      (("num_reclassifications", "Reclassification"), ("num_categories", "Number Of Categories"))}
        for block in self.stats.values():
            stats.addWidget(block)
        lay.addLayout(stats)

        charts = QHBoxLayout()
        charts.setSpacing(12)
        self.live_reclassification = self._live_chart(charts, "Reclassification")
        self.live_categories = self._live_chart(charts, "Categories")
        lay.addLayout(charts, 1)
        return page

    @staticmethod
    def _live_chart(parent, title):
        card = Card()
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 12, 16, 10)
        cl.setSpacing(6)
        cl.addWidget(label(title, "CardTitle"))
        chart = ChartWidget(compact=False)
        cl.addWidget(chart, 1)
        parent.addWidget(card, 1)
        return chart

    def _show_paths(self):
        cfg = self.state.config
        for lbl, path in ((self.input_lbl, cfg.input_path), (self.output_lbl, cfg.output_path)):
            lbl.setToolTip(path)
            lbl.setText(lbl.fontMetrics().elidedText(path, Qt.ElideMiddle, 340))

    # ---- actions -----------------------------------------------------------

    def _start(self):
        text = self.duration.text().strip()
        if not text.isdigit() or not 1 <= int(text) <= MAX_SECONDS:
            self.error.setText(f"Enter a whole number of seconds between 1 and {MAX_SECONDS}.")
            self.error.show()
            return
        self.error.hide()
        self.trainer.start(int(text))

    def _rename(self, name):
        if self._run:
            runs.rename(self._run, name)
            self.state.runs_changed.emit()

    def _refresh_name(self):
        # the run may have been renamed from the Graphs page
        if self._run and self._run.get("path"):
            try:
                name = runs.load(self._run["path"])["name"]
            except (OSError, ValueError):
                return
            if name != self._run["name"]:
                self._run["name"] = name
                self.header.set_name(name)

    # ---- trainer signals --------------------------------------------------

    def _set_running(self, running):
        self.start_btn.setVisible(not running)
        self.stop_btn.setVisible(running)
        self.duration.setEnabled(not running)
        self.progress_box.setVisible(running)

    def _on_started(self, run):
        self._live = {"x": [], "num_reclassifications": [], "num_categories": []}
        self._set_running(True)
        self.progress.setValue(0)
        self.progress_lbl.setText("Starting training")
        for block in self.stats.values():
            block.set_value("–")
        self._update_live_charts()
        self.body.setCurrentIndex(LIVE)

    def _on_progress(self, fraction, elapsed):
        self.progress.setValue(round(fraction * 1000))
        done = len(self._live["x"])
        remaining = elapsed / fraction - elapsed if fraction > 0 else 0
        self.progress_lbl.setText(
            f"Iteration {done} ·  {elapsed:.0f} s elapsed  ·  ~{remaining:.0f} s left")

    def _on_iteration(self, iteration, metrics):
        self._live["x"].append(iteration)
        for key, value in metrics.items():
            self._live[key].append(value)
        self.stats["num_reclassifications"].set_value(f"{metrics['num_reclassifications']}")
        self.stats["num_categories"].set_value(f"{metrics['num_categories']}")
        self._update_live_charts()

    def _update_live_charts(self):
        lv = self._live
        for chart, key, title, fmt in ((self.live_reclassification, "num_reclassifications", "Number Of Reclassifications", "int"),
                                       (self.live_categories, "num_categories", "Number Of Categories", "int")):
            chart.set_spec({
                "kind": "line", "x": lv["x"], "x_label": "Iteration",
                "y_label": title, "y_format": fmt, "empty": "Waiting for the first iteration…",
                "series": [{"name": title, "slot": 0, "values": lv[key]}],
            })

    def _on_finished(self, run):
        self._set_running(False)
        self._run = run
        self.header.set_run(run)
        self.results.set_charts(run["charts"])
        self.body.setCurrentIndex(RESULTS)

    def _on_idle(self, message):
        self._set_running(False)
        self.empty.setText(message)
        self.body.setCurrentIndex(EMPTY)

    def _on_failed(self, message):
        self._on_idle("Run failed.")
        self.error.setText(message)
        self.error.show()
