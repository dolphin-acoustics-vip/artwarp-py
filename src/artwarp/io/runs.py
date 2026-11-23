"""
Saved runs on disk, shared by the CLI and the GUI.

run.json      metadata
results.pkl   full TrainingResults

@author: Jaime Barquero Cebrian
"""
from datetime import datetime
import json
from pathlib import Path

from artwarp.io.exporters import export_results

RUN_FILE = "run.json"
RESULTS_FILE = "results.pkl"

def list_runs(root):
    """All runs under root, newest first. Invalid run.json are skipped."""
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
    """Read a run.json"""
    run = json.loads((Path(path) / RUN_FILE).read_text())
    # path added for callers
    run["path"] = str(path)
    return run

def new_run(network_params, data_params, input_path, output_path, name=None):
    """
    Run dict before training

    Parameters:
        network_params: arguments for ARTwarp network.
        data_params: tempres, resample, sample_interval, max_contour_length.
        name defaults to the timestamp; id is the same in compact form.
    """
    now = datetime.now()
    run = {
        "id": now.strftime("%Y%m%d-%H%M%S"),
        "name": name or now.strftime("%Y-%m-%d %H:%M:%S"),
        "created": now.isoformat(timespec="seconds"),
        "network_parameters": network_params,
        "data_parameters": data_params,
        "input_path": input_path,
        "output_path": output_path,
    }
    return run


def save_run(run, results, names):
    """ Adds remaining information to run dictionary"""

    run["summary"] = {"number_categories": int(results.num_categories),
                              "number_iterations": int(results.num_iterations),
                              "converged": bool(results.converged),
                              "training_time": float(results.training_time)}

    root = Path(run["output_path"])
    root.mkdir(parents=True, exist_ok=True)
    folder, n = root / run["id"], 2
    # Name of folder is id, or id-2, -3, ... if duplicate exists.
    while folder.exists():
        folder = root / f"{run['id']}-{n}"
        n += 1
    run["id"] = folder.name
    run["path"] = str(folder)

    export_results(results, str(Path(run["path"])/RESULTS_FILE), contour_names=names)
    _write(run)
    return run


def rename(run, name):
    """Changes name and run.json."""
    run["name"] = name
    _write(run)


def _write(run):
    """Write run.json; path is removed from run.json."""
    folder = Path(run["path"])
    folder.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in run.items() if k != "path"}
    (folder / RUN_FILE).write_text(json.dumps(data, indent=1))
