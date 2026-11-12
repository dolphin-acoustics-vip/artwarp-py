"""User settings (input/output paths), stored as JSON inside the project."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SETTINGS_FILE = PROJECT_ROOT / "data" / "settings.json"
DEFAULTS = {
    "input_path": str(PROJECT_ROOT / "examples" / "data"),
    "output_path": str(PROJECT_ROOT / "data" / "runs"),
}


@dataclass
class Config:
    input_path: str
    output_path: str

    @classmethod
    def load(cls):
        data = dict(DEFAULTS)
        try:
            data.update(json.loads(SETTINGS_FILE.read_text()))
        except (OSError, ValueError):
            pass
        return cls(input_path=data["input_path"], output_path=data["output_path"])

    def save(self):
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2))
