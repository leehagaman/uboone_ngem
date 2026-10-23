#!/bin/bash
# Four BDT trainings comparing training choices (compare them with src/compare_bdt_trainings.py):
#   {weighted (normalized), unweighted (unit) training events} x {XGBoost default, tuned hyperparameters}
# The tuned hyperparameters are the best trials of the 2026-10-20 tuning studies, one per weight mode.
# The trainings run one at a time (each uses all cores); the chain stops at the first failure.
#
# Usage (from anywhere; runs for a while, so usually in the background):
#   nohup bash run_bdt_comparison_trainings.sh [tag] > bdt_comparison_nohup.out 2>&1 &
# tag defaults to <YYYY_MM_DD>; the trainings are saved as training_outputs/bdt_cmp_<weights>_<params>_<tag>,
# each logging to train_bdt_cmp_<weights>_<params>_<tag>_nohup.out.

TAG="${1:-$(date +%Y_%m_%d)}"

cd "$(dirname "$0")" || exit 1
source ../uv_base/bin/activate || exit 1

NORMALIZED_PARAMS=hyperparameter_tuning_outputs/hp_normalized_2026_10_20b_best_params.json
UNIT_PARAMS=hyperparameter_tuning_outputs/hp_unit_2026_10_20b_best_params.json

start_all=$SECONDS
train() {
    local name=$1; shift
    echo "[$(date '+%F %T')] starting: $name"
    local t0=$SECONDS
    python -u src/train.py --name "$name" "$@" > "train_${name}_nohup.out" 2>&1 \
        || { echo "[$(date '+%F %T')] FAILED: $name (see train_${name}_nohup.out)"; exit 1; }
    echo "[$(date '+%F %T')] finished: $name ($(( (SECONDS - t0) / 60 )) min)"
}

train "bdt_cmp_weighted_default_$TAG" --weighted
train "bdt_cmp_weighted_tuned_$TAG" --weighted --params_json "$NORMALIZED_PARAMS"
train "bdt_cmp_unweighted_default_$TAG"
train "bdt_cmp_unweighted_tuned_$TAG" --params_json "$UNIT_PARAMS"

echo "[$(date '+%F %T')] all trainings done ($(( (SECONDS - start_all) / 60 )) min total)"
echo "compare with: python src/compare_bdt_trainings.py --tag $TAG"
