#!/bin/bash
# Learning rate scan: BDT trainings with fixed tuned hyperparameters (from hyperparameter_tuning.py, which tunes
# at one fixed learning rate) at several learning rates. Lower learning rates need roughly proportionally more
# boosting rounds (and time); the scan shows where the validation loss stops improving. The trainings run one at
# a time (each uses all cores), fastest first; the chain stops at the first failure.
#
# Usage (from anywhere; runs for many hours, so usually in the background):
#   nohup bash run_learning_rate_scan.sh <study_name>_best_params.json [tag] > lr_scan_nohup.out 2>&1 &
# tag defaults to <YYYY_MM_DD>; the trainings are saved as training_outputs/lr_scan_<learning rate>_<tag>,
# each logging to train_lr_scan_<learning rate>_<tag>_nohup.out. Compare them with src/compare_bdt_trainings.py.

if [ $# -lt 1 ] || [ $# -gt 2 ] || [ ! -f "$1" ]; then
    echo "Usage: $0 <study_name>_best_params.json [tag]"
    exit 1
fi
PARAMS_JSON="$(realpath "$1")"
TAG="${2:-$(date +%Y_%m_%d)}"

LEARNING_RATES=(0.3 0.15 0.1 0.05 0.03)

cd "$(dirname "$0")" || exit 1
source ../uv_base/bin/activate || exit 1

start_all=$SECONDS
names=()
for lr in "${LEARNING_RATES[@]}"; do
    name="lr_scan_${lr}_$TAG"
    echo "[$(date '+%F %T')] starting: $name"
    t0=$SECONDS
    python -u src/train.py --name "$name" --params_json "$PARAMS_JSON" --learning_rate "$lr" > "train_${name}_nohup.out" 2>&1 \
        || { echo "[$(date '+%F %T')] FAILED: $name (see train_${name}_nohup.out)"; exit 1; }
    echo "[$(date '+%F %T')] finished: $name ($(( (SECONDS - t0) / 60 )) min)"
    names+=("$name")
done

echo "[$(date '+%F %T')] all trainings done ($(( (SECONDS - start_all) / 60 )) min total)"
echo "compare with: python src/compare_bdt_trainings.py --output_name lr_scan_$TAG --trainings ${names[*]}"
