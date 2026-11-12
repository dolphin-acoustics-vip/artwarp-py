"""
Runs the training on a Worker thread, allows training and GUI be on separate threads.
"""
import time
from datetime import datetime
from pathlib import Path
from functools import partial
from artwarp.io.exporters import export_results



from PySide6.QtCore import QObject, QThread, Signal, Slot
from artwarp import ARTwarp, load_contours

from . import mock_data, runs

LIVE_KEYS = ("train_loss", "val_loss", "train_acc", "val_acc")


class _Worker(QObject):
    tick = Signal(float)        # seconds elapsed
    epoch_done = Signal(int)    # 1-based epoch
    done = Signal(bool, object, object)         # True = completed, False = stopped
    failed = Signal(str)

    def __init__(self, duration, epochs, data):
        super().__init__()
        self.duration = duration
        self.data = data
        self.epochs = epochs
        self._stop = False

    @Slot()
    def run(self):
        try:
            contours, names = load_contours(self.data)
            network = ARTwarp()
            results = network.fit(contours, names)
            self.done.emit(True, results, names)
        except Exception as e:
            self.failed.emit(str(e))

    def stop(self):
        self._stop = True


class Trainer(QObject):
    started = Signal(dict)            # run metadata (no charts yet)
    progress = Signal(float, float)   # fraction done, seconds elapsed
    epoch = Signal(int, dict)         # epoch, live metrics for that epoch
    finished = Signal(dict)           # saved run, with charts
    stopped = Signal()
    failed = Signal(str)

    def __init__(self, state):
        super().__init__(state)
        self.state = state
        self._thread = None
        self._worker = None
        self._run = None
        self._curves = None

    @property
    def running(self):
        return self._run is not None

    def start(self, duration):
        if self.running:
            return
        if self._thread is not None:
            self._thread.wait()

        now = datetime.now()
        seed = int(now.timestamp() * 1000) % 2**31
        epochs = max(5, min(50, round(duration)))
        self._curves = mock_data.make_curves(seed, epochs)
        self._run = {
            "id": now.strftime("%Y%m%d-%H%M%S"),
            "name": now.strftime("%Y-%m-%d %H:%M:%S"),
            "created": now.isoformat(timespec="seconds"),
            "duration_s": duration,
            "epochs": epochs,
            "seed": seed,
            "input_path": self.state.config.input_path,
            "output_path": self.state.config.output_path,
        }

        thread = QThread(self)
        worker = _Worker(duration, epochs, self.state.config.input_path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.tick.connect(self._on_tick)
        worker.epoch_done.connect(self._on_epoch)
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

    @Slot(float)
    def _on_tick(self, elapsed):
        if self._run:
            self.progress.emit(min(1.0, elapsed / self._run["duration_s"]), elapsed)

    @Slot(int)
    def _on_epoch(self, epoch):
        if self._run:
            self.epoch.emit(epoch, {k: self._curves[k][epoch - 1] for k in LIVE_KEYS})

    @Slot(bool, object, object)
    def _on_done(self, completed, results, names):
        run, self._run = self._run, None
        if not completed:
            self.stopped.emit()
            return
        run["summary"] = mock_data.summarise(self._curves)
        run["charts"] = mock_data.build_charts(run["seed"], run["epochs"], self._curves)
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
