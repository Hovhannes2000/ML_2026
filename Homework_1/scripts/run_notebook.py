"""Выполнить notebook локальным Python; сохранять прогресс после каждой ячейки."""

from pathlib import Path
import json
import os
import sys
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
runtime = ROOT / "tmp" / "jupyter"
kernel = runtime / "kernels" / "python3"
kernel.mkdir(parents=True, exist_ok=True)
(runtime / "runtime").mkdir(exist_ok=True)
spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python 3 (local)", "language": "python"}
(kernel / "kernel.json").write_text(json.dumps(spec))
os.environ["JUPYTER_PATH"] = str(runtime)
os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime / "runtime")
os.environ["IPYTHONDIR"] = str(ROOT / "tmp" / "ipython")
os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp" / "matplotlib")
path = ROOT / "experiments.ipynb"
notebook = nbformat.read(path, as_version=4)


def save_progress(cell=None, cell_index=None, **kwargs):
    nbformat.write(notebook, path)
    if cell_index is not None:
        print(f"Выполнена ячейка {cell_index + 1}/{len(notebook.cells)}", flush=True)


client = NotebookClient(notebook, timeout=7200, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}}, on_cell_executed=save_progress)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print("Ноутбук выполнен и сохранён.")
