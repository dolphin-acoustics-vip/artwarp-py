from PySide6.QtCore import QElapsedTimer, QTimer, Qt
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QHBoxLayout, QProgressBar, QPushButton, QSpinBox, QStackedWidget,
                               QVBoxLayout, QWidget)

from .. import icons, runs
from ..widgets.chart import ChartWidget
from ..widgets.common import Card, PageHeader, RunHeader, StatBlock, label
from ..widgets.results import ResultsView

def _path_label(title, parent):
    col = QVBoxLayout()
    col.setSpacing(4)
    col.addWidget(label(title, "FieldLabel"))
    value = label("", "Subtle")
    col.addWidget(value)
    parent.addLayout(col, 1)
    return value

def _field(title, widget, parent):
    container = QWidget()
    box = QVBoxLayout(container)
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(4)
    title_label = label(title, "FieldLabel")
    box.addWidget(title_label)
    container.title_label = title_label
    box.addWidget(widget)
    parent.addWidget(container, 1)
    return container

class TrainingPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.trainer = state.trainer
        self._run = None
        self._live_iteration = {}
        self._timer = QElapsedTimer()
        self._timer_ticker = QTimer()
        self._timer_ticker.setInterval(1000)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)
        root.addWidget(PageHeader("Training", "All parameters are editable. Currently they show default values."))
        root.addWidget(self._build_controls())

        self.body = QStackedWidget()
        self.body.addWidget(self._build_live_iteration())
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

        self._timer_ticker.timeout.connect(self._update_progress)
        self.trainer.started.connect(self._on_started)
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
        
        self.vigilance = QDoubleSpinBox()
        self.vigilance.setRange(1, 99)
        self.vigilance.setValue(85.0)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(1e-2,1)
        self.learning_rate.setDecimals(3)
        self.learning_rate.setSingleStep(0.05)
        self.learning_rate.setValue(0.1)
        self.bias = QDoubleSpinBox()
        self.bias.setRange(0,1)
        self.bias.setDecimals(3)
        self.bias.setSingleStep(0.05)
        self.bias.setValue(0.0)
        self.max_categories = QSpinBox()
        self.max_categories.setRange(1,100)
        self.max_categories.setValue(100)
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(1,99)
        self.max_iterations.setValue(50)
        self.warp_factor_level = QSpinBox()
        self.warp_factor_level.setRange(2,10)
        self.warp_factor_level.setValue(3)
        self.random_seed = QSpinBox()
        self.random_seed.setRange(0,1e7)
        self.random_seed.setSpecialValueText("Random")
        self.randomize_sample_order = QCheckBox("Randomise Sample Order")
        self.randomize_sample_order.setChecked(True)
        self.verbose = QCheckBox("Verbose (CLI - Terminal)")
        self.recat_single_categories = QCheckBox("Recategorise Single Categories")
        self.compare_warped = QCheckBox("Compare Warped")
        self.deprioritize_lone_category_search = QCheckBox("Deprioritize Lone Category Search")
        self.purge_empty_categories = QCheckBox("Purge Empty Categories")

        self.tempres = QDoubleSpinBox()
        self.tempres.setRange(0.01,1)
        self.tempres.setDecimals(3)
        self.tempres.setSingleStep(0.01)
        self.tempres.setValue(0.01)
        self.sample_interval = QDoubleSpinBox()
        self.sample_interval.setRange(0.01,1)
        self.sample_interval.setDecimals(3)
        self.sample_interval.setSingleStep(0.01)
        self.sample_interval.setValue(0.01)
        self.resample = QCheckBox("Resample")
        self.resample.setChecked(True)
        self.max_contour_length = QSpinBox()
        self.max_contour_length.setRange(0,1e7)
        self.max_contour_length.setSpecialValueText("No Cap")

        self.param_box = QWidget()
        form = QVBoxLayout(self.param_box)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)

        fields = QHBoxLayout()
        fields.setSpacing(24)
        _field("Vigilance", self.vigilance, fields)
        _field("Learning rate", self.learning_rate, fields)
        _field("Bias", self.bias, fields)
        _field("Max categories", self.max_categories, fields)
        _field("Max iterations", self.max_iterations, fields)
        _field("Warp factor", self.warp_factor_level, fields)
        _field("Random seed", self.random_seed, fields)
        form.addLayout(fields)

        data_fields = QHBoxLayout()
        data_fields.setSpacing(24)
        tempres_field = _field("Temporal Resolution", self.tempres, data_fields)
        interval_field = _field("Sample Interval", self.sample_interval, data_fields)
        
        def resample_toggled(on):
            for field, title in ((tempres_field, "Temporal Resolution"),
                                    (interval_field, "Sample Interval")):
                field.setEnabled(on)
                field.title_label.setText(title if on else f"{title} (needs Resample)")
        self.resample.toggled.connect(resample_toggled)

        _field("Maximum Contour Length", self.max_contour_length, data_fields)
        form.addLayout(data_fields)

        checks_line1 = QHBoxLayout()
        checks_line1.setSpacing(24)
        checks_line1.addWidget(self.resample)
        checks_line1.addWidget(self.randomize_sample_order)
        checks_line1.addWidget(self.verbose)
        checks_line1.addWidget(self.recat_single_categories)
        checks_line1.addStretch(1)
        form.addLayout(checks_line1)

        checks_line2 = QHBoxLayout()
        checks_line2.setSpacing(24)
        checks_line2.addWidget(self.deprioritize_lone_category_search)
        checks_line2.addWidget(self.purge_empty_categories)
        checks_line2.addWidget(self.compare_warped)
        checks_line2.addStretch(1)
        form.addLayout(checks_line2)
        

        lay.addWidget(self.param_box)

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

    def _build_live_iteration(self):
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
        self.live_reclassification = self._live_iteration_chart(charts, "Reclassification")
        self.live_categories = self._live_iteration_chart(charts, "Categories")
        lay.addLayout(charts, 1)
        return page

    @staticmethod
    def _live_iteration_chart(parent, title):
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
        network_params = {
            "vigilance": self.vigilance.value(),
            "learning_rate": self.learning_rate.value(),
            "bias": self.bias.value(),
            "max_categories": self.max_categories.value(),
            "max_iterations": self.max_iterations.value(),
            "warp_factor_level": self.warp_factor_level.value(),
            "random_seed": self.random_seed.value(),
            "randomize_sample_order": self.randomize_sample_order.isChecked(),
            "verbose": self.verbose.isChecked(),
            "recat_single_categories": self.recat_single_categories.isChecked(),
            "compare_warped": self.compare_warped.isChecked(),
            "deprioritize_lone_category_search": self.deprioritize_lone_category_search.isChecked(),
            "purge_empty_categories": self.purge_empty_categories.isChecked(),
        }
        data_params = {
            "tempres": self.tempres.value(),
            "resample": self.resample.isChecked(),
            "sample_interval": self.sample_interval.value(),
            "max_contour_length": self.max_contour_length.value(),
        }
        if network_params["random_seed"] == 0:
            network_params["random_seed"] = None
        if data_params["max_contour_length"] == 0:
            data_params["max_contour_length"] = None
        self.error.hide()
        self.trainer.start(network_params, data_params)

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
        self.param_box.setEnabled(not running)
        self.progress_box.setVisible(running)
        if not running: self._timer_ticker.stop()

    def _on_started(self, run):
        self._live_iteration = {"x": [], "num_reclassifications": [], "num_categories": []}
        self._set_running(True)
        self.progress.setValue(0)
        self._timer.start()
        self._timer_ticker.start()
        self.progress.setRange(0, run["network_parameters"]["max_iterations"])
        self.progress_lbl.setText("Starting training")
        for block in self.stats.values():
            block.set_value("–")
        self._update_live_iteration_charts()

    def _on_iteration(self, iteration, metrics):
        self._live_iteration["x"].append(iteration)
        self.progress.setValue(iteration)
        for key, value in metrics.items():
            self._live_iteration[key].append(value)
        self.stats["num_reclassifications"].set_value(f"{metrics['num_reclassifications']}")
        self.stats["num_categories"].set_value(f"{metrics['num_categories']}")
        self._update_progress()
        self._update_live_iteration_charts()

    def _update_live_iteration_charts(self):
        lv = self._live_iteration
        for chart, key, title, fmt in ((self.live_reclassification, "num_reclassifications", "Number Of Reclassifications", "int"),
                                       (self.live_categories, "num_categories", "Number Of Categories", "int")):
            chart.set_spec({
                "kind": "line", "x": lv["x"], "x_label": "Iteration",
                "y_label": title, "y_format": fmt, "empty": "Waiting for the first iteration…",
                "series": [{"name": title, "slot": 0, "values": lv[key]}],
            })

    def _update_progress(self):
        self.progress_lbl.setText(f"Training... · Iteration {len(self._live_iteration['x'])} · Elapsed time {self._timer.elapsed()/1000.0:.2f}s")

    def _on_finished(self, run):
        self._set_running(False)
        self._run = run
        self.header.set_run(run)
        self.results.set_charts(run["charts"])

    def _on_idle(self, message):
        self._set_running(False)
        self.empty.setText(message)

    def _on_failed(self, message):
        self._on_idle("Run failed.")
        self.error.setText(message)
        self.error.show()
