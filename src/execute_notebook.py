"""Execute a notebook in place, the same as "Run All" in Jupyter, saving the outputs into the notebook.

The notebook runs with ipynb_notebooks/ (the notebook's own directory) as its working directory, so its
relative paths behave as in Jupyter.  If a cell raises, the notebook is still saved with every output up to
and including the failing cell's traceback, and this script exits nonzero.

Usage:
    python src/execute_notebook.py ipynb_notebooks/open_data_distributions.ipynb
"""
import argparse
import os
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def format_duration(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute a notebook in place, saving its outputs.")
    parser.add_argument("notebook", help="path to the .ipynb file")
    args = parser.parse_args()

    nb_path = os.path.abspath(args.notebook)
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=None,  # no per-cell timeout: the systematics cells can take hours
        kernel_name="python3",
        resources={"metadata": {"path": os.path.dirname(nb_path)}},
    )

    print(f"executing {nb_path} ({len(nb.cells)} cells)...", flush=True)
    start_time = time.time()
    try:
        client.execute()
    except CellExecutionError as e:
        nbformat.write(nb, nb_path)
        print(f"FAILED after {format_duration(time.time() - start_time)}; saved partial outputs to {nb_path}")
        print(e)
        raise SystemExit(1)
    nbformat.write(nb, nb_path)
    print(f"done in {format_duration(time.time() - start_time)}; saved outputs to {nb_path}")
