#!/bin/bash
# Full re-processing chain:
#   1. create_df.py (file-level dfs + merge)
#   2. create_rw_syst_df.py and create_detvar_df.py, in parallel
#   3. train.py (skipped with --no_train)
# Each step logs to the same *_nohup.out file as the README commands. The chain stops
# at the first failed step.
#
# Usage (from anywhere; runs for several hours, so usually in the background):
#   nohup bash run_full_reprocessing.sh [--no_train] [training_name] > reprocess_nohup.out 2>&1 &
# training_name defaults to all_vars_<YYYY_MM_DD>.

DO_TRAIN=1
TRAIN_NAME=""
for arg in "$@"; do
    case "$arg" in
        --no_train) DO_TRAIN=0 ;;
        -*) echo "unknown option: $arg"; exit 1 ;;
        *)
            [ -z "$TRAIN_NAME" ] || { echo "more than one training_name given: $TRAIN_NAME, $arg"; exit 1; }
            TRAIN_NAME="$arg" ;;
    esac
done
TRAIN_NAME="${TRAIN_NAME:-all_vars_$(date +%Y_%m_%d)}"

cd "$(dirname "$0")" || exit 1
source ../uv_base/bin/activate || exit 1

start_all=$SECONDS
step_start() { echo "[$(date '+%F %T')] starting: $1"; step_t0=$SECONDS; }
step_done()  { echo "[$(date '+%F %T')] finished: $1 ($(( (SECONDS - step_t0) / 60 )) min)"; }
fail()       { echo "[$(date '+%F %T')] FAILED: $1 (see $2)"; exit 1; }

if [ $DO_TRAIN -eq 1 ]; then
    echo "training name: $TRAIN_NAME"
else
    echo "--no_train: skipping BDT training"
fi

step_start "create_df"
python -u src/create_df.py -m --create_file_dfs --merge_file_dfs > create_file_dfs_nohup.out 2>&1 \
    || fail "create_df" create_file_dfs_nohup.out
step_done "create_df"

# rw needs presel_df_train_vars.parquet from create_df (for the derived coherent-1g rows);
# detvar is independent. Their intermediate file names don't overlap.
step_start "create_rw_syst_df + create_detvar_df (parallel)"
python -u src/create_rw_syst_df.py -m > weights_nohup.out 2>&1 &
rw_pid=$!
python -u src/create_detvar_df.py -m > detvar_nohup.out 2>&1 &
detvar_pid=$!
wait $rw_pid;     rw_status=$?
wait $detvar_pid; detvar_status=$?
[ $rw_status -eq 0 ]     || fail "create_rw_syst_df" weights_nohup.out
[ $detvar_status -eq 0 ] || fail "create_detvar_df" detvar_nohup.out
step_done "create_rw_syst_df + create_detvar_df (parallel)"

if [ $DO_TRAIN -eq 1 ]; then
    step_start "train"
    python -u src/train.py --name "$TRAIN_NAME" > train_nohup.out 2>&1 \
        || fail "train" train_nohup.out
    step_done "train"
fi

echo "[$(date '+%F %T')] all steps done ($(( (SECONDS - start_all) / 60 )) min total)"
