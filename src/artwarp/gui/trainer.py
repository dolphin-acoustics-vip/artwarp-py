"""
Runs the training on a Worker thread, allows training and GUI be on separate threads.
"""
from datetime import datetime
from pathlib import Path
from functools import partial
from artwarp.io.exporters import export_results
import threading

from PySide6.QtCore import QObject, QThread, Signal, Slot
from artwarp import ARTwarp, load_contours

from . import charts, runs


class _Worker(QObject):
    iteration_done = Signal(object)
    done = Signal(bool, object, object)
    failed = Signal(str)

    def __init__(self, params, data):
        super().__init__()
        self.parameters = params
        self.data = data
        self._stop = threading.Event()
        self._callback_stop = False

    @Slot()
    def run(self):
        try:
            contours, names = load_contours(self.data)
            network = ARTwarp(**self.parameters)
            results = network.fit(contours, names, on_iteration=self.callback)
            self.done.emit(not self._callback_stop, results, names)
        except Exception as e:
            self.failed.emit(str(e))

    def callback(self, snapshot):
        self.iteration_done.emit(snapshot)
        if self._stop.is_set():
            self._callback_stop = True
            return True

    def stop(self):
        self._stop.set()


class Trainer(QObject):
    started = Signal(dict)            # run metadata (no charts yet)
    iteration = Signal(int, dict)         # iteration, live metrics for that iteration
    finished = Signal(dict)           # saved run, with charts
    stopped = Signal()
    failed = Signal(str)

    def __init__(self, state):
        super().__init__(state)
        self.state = state
        self._thread = None
        self._worker = None
        self._run = None

    @property
    def running(self):
        return self._run is not None

    def start(self, params):
        if self.running:
            return
        if self._thread is not None:
            self._thread.wait()

        now = datetime.now()
        self._run = {
            "id": now.strftime("%Y%m%d-%H%M%S"),
            "name": now.strftime("%Y-%m-%d %H:%M:%S"),
            "created": now.isoformat(timespec="seconds"),
            "parameters": params,
            "input_path": self.state.config.input_path,
            "output_path": self.state.config.output_path,
        }

        thread = QThread(self)
        worker = _Worker(params, self.state.config.input_path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.iteration_done.connect(self._on_iteration)
        worker.done.connect(self._on_done)
        worker.done.connect(thread.quit)
        worker.failed.connect(self._on_fail)
        worker.failed.connect(thread.quit)
        thread.finished.connect(partial(self._on_thread_finished, thread))
        self._thread, self._worker = thread, worker
        thread.start()
        self.started.emit(dict(self._run))

    def stop(self):
        if self._worker is not None:
            self._worker.stop()

    def shutdown(self):
        self.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)

    @Slot(object)
    def _on_iteration(self, snapshot):
        if self._run:
            dic = { "num_reclassifications": snapshot.num_reclassifications, "num_categories": snapshot.num_categories}
            self.iteration.emit(snapshot.iteration, dic)

    @Slot(bool, object, object)
    def _on_done(self, completed, results, names):
        run, self._run = self._run, None
        if not completed:
            self.stopped.emit()
            return
        run["summary"] = {"number_categories": int(results.num_categories),
                          "number_iterations": int(results.num_iterations),
                          "converged": bool(results.converged),
                          "training_time": float(results.training_time)}
        run["charts"] = charts.build_charts(results)
        try:
            runs.save(run, run["output_path"])
            export_results(results, str(Path(run["path"])/runs.RESULTS_FILE), contour_names=names)
        except OSError as e:
            self.failed.emit(f"Couldn't save run to {run['output_path']}: {e.strerror or e}")
            return
        self.finished.emit(run)
        self.state.runs_changed.emit()

    @Slot(str)
    def _on_fail(self, msg):
        self._run = None
        self.failed.emit(f"{msg}")

    def _on_thread_finished(self, thread):
        # a newer run may already own self._thread, so only clear our own refs
        thread.deleteLater()
        if thread is self._thread:
            self._thread = None
            self._worker = None
