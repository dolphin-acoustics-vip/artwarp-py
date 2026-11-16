"""
User specified settings. Shows current input and output path.
Edits state.config and persists to SETTINGS_FILE, setting CLI and GUI defaults.

@author: Jaime Barquero Cebrian
"""

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from artwarp.gui import icons, theme
from artwarp.io.config import DEFAULTS, SETTINGS_FILE
from artwarp.gui.widgets.common import Card, PageHeader, label, open_folder, set_role


class SettingsPage(QWidget):
    """ Emits config_changed. Only emits runs_changed if output folder changed"""
    def __init__(self, state):
        super().__init__()
        self.state = state

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)
        root.addWidget(PageHeader("Settings", "Where training reads its data and writes its runs."))

        card = Card()
        card.setMaximumWidth(780)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 18, 20, 18)
        cl.setSpacing(18)
        self.input = self._path_row(cl, "Input data folder",
                                    "Read by the trainer when a run starts.")
        divider = QFrame()
        divider.setObjectName("Divider")
        divider.setFixedHeight(1)
        cl.addWidget(divider)
        self.output = self._path_row(cl, "Output folder",
                                     "Each run is saved here as <folder>/<id>. Graphs lists the runs in this folder.")

        footer = QHBoxLayout()
        self.status = label("", "Muted")
        footer.addWidget(self.status, 1)
        reset = QPushButton("Restore defaults")
        reset.setObjectName("Ghost")
        reset.clicked.connect(self._reset)
        self.save_btn = QPushButton("Save changes")
        self.save_btn.setObjectName("Primary")
        self.save_btn.clicked.connect(self._save)
        footer.addWidget(reset)
        footer.addWidget(self.save_btn)
        cl.addLayout(footer)
        root.addWidget(card)
        root.addWidget(label(f"Settings file: {SETTINGS_FILE}", "Muted"))
        root.addStretch()

        self._load()

    def _path_row(self, parent, title, desc):
        """
        Builds title, description and row of QLineEdit. Includes Browse and Reveal.
        Returns QLineEdit.
        """

        # Negative spacing pulls description up under the title
        parent.addWidget(label(title, "FieldTitle"))
        parent.addSpacing(-12)
        parent.addWidget(label(desc, "Muted", wrap=True))
        parent.addSpacing(-8)
        row = QHBoxLayout()
        row.setSpacing(8)
        edit = QLineEdit()
        edit.textChanged.connect(self._on_edit)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse(edit))
        reveal = QPushButton()
        reveal.setObjectName("Ghost")
        reveal.setToolTip("Open in Finder")
        reveal.setIcon(icons.icon("folder", theme.TEXT_2))
        # Reveal only when the folder exists
        reveal.clicked.connect(lambda: Path(edit.text()).is_dir() and open_folder(edit.text()))
        row.addWidget(edit, 1)
        row.addWidget(browse)
        row.addWidget(reveal)
        parent.addLayout(row)
        return edit

    def _load(self):
        self.input.setText(self.state.config.input_path)
        self.output.setText(self.state.config.output_path)
        self._on_edit()

    def _browse(self, edit):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text())
        if folder:
            edit.setText(folder)

    def _reset(self):
        """Only fills fields, does not save anything"""
        self.input.setText(DEFAULTS["input_path"])
        self.output.setText(DEFAULTS["output_path"])

    def _values(self):
        """Grabs current input and output paths"""
        # Expanduser allows ~ for Paths
        return (str(Path(self.input.text().strip()).expanduser()) if self.input.text().strip() else "",
                str(Path(self.output.text().strip()).expanduser()) if self.output.text().strip() else "")

    def _on_edit(self):
        """Saved only enabled when both input and output fields are filled and different from saved config"""
        inp, out = self._values()
        cfg = self.state.config
        self.save_btn.setEnabled(bool(inp and out) and (inp, out) != (cfg.input_path, cfg.output_path))
        self._message("")

    def _message(self, text, error=False):
        set_role(self.status, "Error" if error else "Muted")
        self.status.setText(text)

    def _save(self):
        """
        Output dir created first (aborts on failure), config object changed and saved
        Allows current run to keep own paths and new run use edited ones.
        """
        inp, out = self._values()
        try:
            Path(out).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._message(f"Can't create output folder: {e.strerror or e}", error=True)
            return
        cfg = self.state.config
        output_changed = out != cfg.output_path
        cfg.input_path, cfg.output_path = inp, out
        cfg.save()
        self.save_btn.setEnabled(False)

        notes = ["Saved."]
        if not Path(inp).is_dir():
            notes.append("Input folder doesn't exist yet.")
        if self.state.trainer.running:
            notes.append("The current run keeps its original paths.")
        self._message(" ".join(notes))
        self.state.config_changed.emit()
        if output_changed:
            self.state.runs_changed.emit()
