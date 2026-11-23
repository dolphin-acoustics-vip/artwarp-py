"""
Runs the training on a Worker thread, allows training and GUI be on separate threads.
Two classes: 
    _Worker - on a QThread during load, resample and network fit
    Trainer - on GUI thread signals _Worker to keep GUI responsive while training

@author: Jaime Barquero Cebrian
"""

from functools import partial
import threading

from PySide6.QtCore import QObject, QThread, Signal, Slot
from artwarp import ARTwarp, load_contours
from artwarp.utils.resample import cap_contour_lengths, prepare_contours

from artwarp.io.runs import save_run, new_run


class _Worker(QObject):
    """Runs on QThread and does load, resample and fit."""
    iteration_done = Signal(object)
    done = Signal(bool, object, object)
    failed = Signal(str)

    def __init__(self, network_params, data_params, data):
        super().__init__()
        self.network_parameters = network_params
        self.data_parameters = data_params
        self.data = data
        self._stop = threading.Event()
        # Records that the stop came from user
        self._callback_stop = False

    @Slot()
    def run(self):
        """ Mirrors command_train CLI"""
        try:
            contours, names, tempres_list = load_contours(self.data, return_tempres=True)
            contours = prepare_contours(self.data_parameters, contours, tempres_list)

            # ARTwarp call
            network = ARTwarp(**self.network_parameters)
            results = network.fit(contours, names, on_iteration=self.callback)
            self.done.emit(not self._callback_stop, results, names)
        except Exception as e:
            self.failed.emit(str(e))

    def callback(self, snapshot):
        """Called by fit every iteration from _Worker thread. Emits snapshot and returns True if fit() should stop"""
        self.iteration_done.emit(snapshot)
        if self._stop.is_set():
            self._callback_stop = True
            return True

    def stop(self):
        self._stop.set()


class Trainer(QObject):
    """
    Runs on GUI thread signals _Worker to keep GUI responsive while training.
    Signals:
        started: carries full new_run dictionary.
        iteration: iteration number and metrics dic (num_reclassifications, num_categories)
        finished: dic of saved run (all metrics in run.json)
        stopped: empty signal
        failed: message

    """
    started = Signal(dict)
    iteration = Signal(int, dict)
    finished = Signal(dict)
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

    def start(self, network_params, data_params):
        """
        Builds _run via new_run before thread starts so run has initial information.
        Creates new QThread per run, connects done/failed to thread.quit
        """
        if self.running:
            return
        if self._thread is not None:
            self._thread.wait()

        self._run = new_run(
            network_params,
            data_params,
            self.state.config.input_path,
            self.state.config.output_path)

        thread = QThread(self)
        worker = _Worker(network_params, data_params, self.state.config.input_path)
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
        """Sets event, takes effect next iteration"""
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
        """ If completed is false run is stopped otherwise calls save_run. TODO: Nothing saved, review to add partial run save
        """
        run, self._run = self._run, None
        if not completed:
            self.stopped.emit()
            return
        try:
            save_run(run, results, names)
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
        """A new run may have replaced references. Only clear references if thread is still current one"""
        # a newer run may already own self._thread, so only clear our own refs
        thread.deleteLater()
        if thread is self._thread:
            self._thread = None
            self._worker = None
