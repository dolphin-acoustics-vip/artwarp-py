"""
Saved runs on disk: artwarp-py/data/runs/
"""
import json
from pathlib import Path

RUN_FILE = "run.json"
RESULTS_FILE = "results.pkl"

def list_runs(root):
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for file in root.glob(f"*/{RUN_FILE}"):
        try:
            run = json.loads(file.read_text())
        except (OSError, ValueError):
            continue
        run["path"] = str(file.parent)
        found.append(run)
    found.sort(key=lambda r: r.get("created", ""), reverse=True)
    return found


def load(path):
    run = json.loads((Path(path) / RUN_FILE).read_text())
    run["path"] = str(path)
    return run


def save(run, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    folder, n = root / run["id"], 2
    while folder.exists():
        folder = root / f"{run['id']}-{n}"
        n += 1
    run["id"] = folder.name
    run["path"] = str(folder)
    _write(run)


def rename(run, name):
    run["name"] = name
    _write(run)


def _write(run):
    folder = Path(run["path"])
    folder.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in run.items() if k != "path"}
    (folder / RUN_FILE).write_text(json.dumps(data, indent=1))
