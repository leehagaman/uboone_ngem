#!/bin/bash
# Runs the distribution notebooks one after another, in place (the outputs are saved into each
# notebook, the same as "Run All" in Jupyter), so they can run in the background without a browser:
#   open_data_distributions, nuwro_fake_data_distributions, gibuu_fake_data_distributions
# Each notebook logs to its own <name>_nohup.out file. A failing notebook is saved with its outputs
# up to the failing cell, and the remaining notebooks still run.
#
# Usage (from anywhere; usually in the background):
#   nohup bash run_distribution_notebooks.sh [notebook_name ...] > notebooks_nohup.out 2>&1 &
# notebook_name is e.g. nuwro_fake_data_distributions (no .ipynb); defaults to all three.
# Close the notebooks in Jupyter first, or saving them there will overwrite the new outputs.

NOTEBOOKS=("$@")
if [ ${#NOTEBOOKS[@]} -eq 0 ]; then
    NOTEBOOKS=(open_data_distributions nuwro_fake_data_distributions gibuu_fake_data_distributions)
fi

cd "$(dirname "$0")" || exit 1
source ../uv_base/bin/activate || exit 1

for nb in "${NOTEBOOKS[@]}"; do
    [ -f "ipynb_notebooks/$nb.ipynb" ] || { echo "no such notebook: ipynb_notebooks/$nb.ipynb"; exit 1; }
done

start_all=$SECONDS
failed=()
for nb in "${NOTEBOOKS[@]}"; do
    echo "[$(date '+%F %T')] starting: $nb"
    t0=$SECONDS
    if python -u src/execute_notebook.py "ipynb_notebooks/$nb.ipynb" > "${nb}_nohup.out" 2>&1; then
        echo "[$(date '+%F %T')] finished: $nb ($(( (SECONDS - t0) / 60 )) min)"
    else
        echo "[$(date '+%F %T')] FAILED: $nb ($(( (SECONDS - t0) / 60 )) min, see ${nb}_nohup.out)"
        failed+=("$nb")
    fi
done

echo "[$(date '+%F %T')] all notebooks done ($(( (SECONDS - start_all) / 60 )) min total)"
if [ ${#failed[@]} -gt 0 ]; then
    echo "failed: ${failed[*]}"
    exit 1
fi
