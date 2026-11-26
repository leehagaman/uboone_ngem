#!/bin/bash
# Full re-processing chain:
#   1. create_df.py (file-level dfs + merge)
#   2. create_rw_syst_df.py and create_detvar_df.py, in parallel (detvar skipped with --no_detvar)
#   3. hyperparameter_tuning.py: a new study hp_<YYYY_MM_DD>, starting from the newest earlier
#      hyperparameter_tuning_outputs/hp_*_best_params.json (~1 day for 100 trials; skipped with --no_hp
#      or --no_train)
#   4. train.py with the tuned hyperparameters (with --no_hp: the newest existing best_params.json),
#      then most_recent_bdt_training_run in src/file_locations.py is set to the new training and
#      visualize_multiclass_probs.ipynb is run on it (skipped with --no_train)
#   5. clear the systematics covariance cache, then run the open data / NuWro / GiBUU distribution
#      notebooks (run_distribution_notebooks.sh), efficiencies.ipynb, plot_many_variables.py, and the p-value summary
#      notebook plot_all_bdt_vars.ipynb (skipped with --no_plots)
# Each step logs to its own *_nohup.out file (the same ones as the README commands). Steps 1-4 stop
# the chain at the first failure; a failing notebook or plot step (the multiclass notebook in step 4,
# everything in step 5) is reported at the end and the remaining ones still run.
#
# Usage (from anywhere; runs for a day or more, so usually in the background):
#   nohup bash run_full_reprocessing.sh [--no_detvar] [--no_hp] [--no_train] [--no_plots] [training_name] \
#       > reprocess_nohup.out 2>&1 &
# training_name defaults to all_vars_<YYYY_MM_DD>. Close the distribution notebooks in Jupyter first, or
# saving them there will overwrite the new outputs.

DO_DETVAR=1
DO_HP=1
DO_TRAIN=1
DO_PLOTS=1
TRAIN_NAME=""
for arg in "$@"; do
    case "$arg" in
        --no_detvar) DO_DETVAR=0 ;;
        --no_hp) DO_HP=0 ;;
        --no_train) DO_TRAIN=0; DO_HP=0 ;;
        --no_plots) DO_PLOTS=0 ;;
        -*) echo "unknown option: $arg"; exit 1 ;;
        *)
            [ -z "$TRAIN_NAME" ] || { echo "more than one training_name given: $TRAIN_NAME, $arg"; exit 1; }
            TRAIN_NAME="$arg" ;;
    esac
done
TODAY="$(date +%Y_%m_%d)"
TRAIN_NAME="${TRAIN_NAME:-all_vars_$TODAY}"
HP_STUDY="hp_$TODAY"
HP_DIR="hyperparameter_tuning_outputs"

cd "$(dirname "$0")" || exit 1
source ../uv_base/bin/activate || exit 1

start_all=$SECONDS
step_start() { echo "[$(date '+%F %T')] starting: $1"; step_t0=$SECONDS; }
step_done()  { echo "[$(date '+%F %T')] finished: $1 ($(( (SECONDS - step_t0) / 60 )) min)"; }
fail()       { echo "[$(date '+%F %T')] FAILED: $1 (see $2)"; exit 1; }
newest_best_params() { ls -t "$HP_DIR"/hp_*_best_params.json 2>/dev/null | head -n 1; }

# ── checks before anything runs, so a naming clash fails now and not a day in ──
if [ $DO_HP -eq 1 ]; then
    [ ! -e "$HP_DIR/$HP_STUDY.db" ] || { echo "$HP_DIR/$HP_STUDY.db already exists (would resume that study)"; exit 1; }
    START_FROM="$(newest_best_params)"
    echo "hyperparameter tuning: new study $HP_STUDY, starting from ${START_FROM:-the XGBoost defaults only}"
fi
if [ $DO_TRAIN -eq 1 ]; then
    # train.py deletes an existing training_outputs/<name>
    [ ! -e "training_outputs/$TRAIN_NAME" ] || { echo "training_outputs/$TRAIN_NAME already exists"; exit 1; }
    if [ $DO_HP -eq 1 ]; then
        PARAMS_JSON="$HP_DIR/${HP_STUDY}_best_params.json"
    else
        PARAMS_JSON="$(newest_best_params)"
        [ -n "$PARAMS_JSON" ] || { echo "--no_hp: no $HP_DIR/hp_*_best_params.json to train with"; exit 1; }
    fi
    echo "training name: $TRAIN_NAME (hyperparameters: $PARAMS_JSON)"
else
    echo "--no_train: skipping hyperparameter tuning and BDT training"
fi
if [ $DO_DETVAR -eq 0 ]; then
    echo "--no_detvar: skipping create_detvar_df (the existing detvar df is left as is)"
fi
if [ $DO_PLOTS -eq 0 ]; then
    echo "--no_plots: skipping the cache clearing, notebooks and plots"
elif [ $DO_TRAIN -eq 0 ]; then
    echo "WARNING: the notebooks and plots will use the existing training" \
         "$(python -c 'import sys; sys.path.insert(0, "src"); from file_locations import most_recent_bdt_training_run as t; print(t)')"
fi

failed=()

step_start "create_df"
python -u src/create_df.py -m --create_file_dfs --merge_file_dfs > create_file_dfs_nohup.out 2>&1 \
    || fail "create_df" create_file_dfs_nohup.out
step_done "create_df"

# rw needs presel_df_train_vars.parquet from create_df (for the derived coherent-1g rows and
# their flux donors); detvar needs coh_1g_reweighting_bins.parquet from create_df (to turn
# the iso1g DetVar rows into NC coherent 1g rows). Their intermediate file names don't overlap.
if [ $DO_DETVAR -eq 1 ]; then
    step_name="create_rw_syst_df + create_detvar_df (parallel)"
else
    step_name="create_rw_syst_df"
fi
step_start "$step_name"
python -u src/create_rw_syst_df.py -m > weights_nohup.out 2>&1 &
rw_pid=$!
if [ $DO_DETVAR -eq 1 ]; then
    python -u src/create_detvar_df.py -m > detvar_nohup.out 2>&1 &
    detvar_pid=$!
fi
wait $rw_pid; rw_status=$?
[ $rw_status -eq 0 ] || fail "create_rw_syst_df" weights_nohup.out
if [ $DO_DETVAR -eq 1 ]; then
    wait $detvar_pid; detvar_status=$?
    [ $detvar_status -eq 0 ] || fail "create_detvar_df" detvar_nohup.out
fi
step_done "$step_name"

if [ $DO_HP -eq 1 ]; then
    step_start "hyperparameter tuning ($HP_STUDY)"
    start_from_args=()
    [ -n "$START_FROM" ] && start_from_args=(--start_from "$START_FROM")
    python -u src/hyperparameter_tuning.py --study_name "$HP_STUDY" "${start_from_args[@]}" \
        --nthread 64 --learning_rate 0.3 --max_boost_rounds 1500 --n_trials 100 > hp_nohup.out 2>&1 \
        || fail "hyperparameter tuning" hp_nohup.out
    [ -f "$PARAMS_JSON" ] || fail "hyperparameter tuning (no $PARAMS_JSON written)" hp_nohup.out
    step_done "hyperparameter tuning ($HP_STUDY)"
fi

if [ $DO_TRAIN -eq 1 ]; then
    step_start "train"
    python -u src/train.py --name "$TRAIN_NAME" --params_json "$PARAMS_JSON" > train_nohup.out 2>&1 \
        || fail "train" train_nohup.out
    sed -i "s/^most_recent_bdt_training_run = .*/most_recent_bdt_training_run = \"$TRAIN_NAME\"/" src/file_locations.py
    grep -q "^most_recent_bdt_training_run = \"$TRAIN_NAME\"$" src/file_locations.py \
        || fail "setting most_recent_bdt_training_run" src/file_locations.py
    echo "set most_recent_bdt_training_run = \"$TRAIN_NAME\" in src/file_locations.py"
    step_done "train"

    step_start "visualize_multiclass_probs notebook"
    python -u src/execute_notebook.py ipynb_notebooks/visualize_multiclass_probs.ipynb > visualize_multiclass_probs_nohup.out 2>&1 \
        || failed+=("visualize_multiclass_probs.ipynb (see visualize_multiclass_probs_nohup.out)")
    step_done "visualize_multiclass_probs notebook"
fi

if [ $DO_PLOTS -eq 1 ]; then
    cache_dir="$(python -c 'import sys; sys.path.insert(0, "src"); from file_locations import covariance_cache_location as c; print(c)')"
    [ -n "$cache_dir" ] && [ -d "$cache_dir" ] || fail "finding the systematics cache directory" "src/file_locations.py"
    n_cached=$(find "$cache_dir" -maxdepth 1 -type f | wc -l)
    find "$cache_dir" -maxdepth 1 -type f -delete
    echo "[$(date '+%F %T')] cleared the systematics cache ($n_cached files in $cache_dir)"

    step_start "distribution notebooks"
    bash run_distribution_notebooks.sh > notebooks_nohup.out 2>&1 || failed+=("distribution notebooks (see notebooks_nohup.out)")
    step_done "distribution notebooks"

    step_start "efficiencies notebook"
    python -u src/execute_notebook.py ipynb_notebooks/efficiencies.ipynb > efficiencies_nohup.out 2>&1 \
        || failed+=("efficiencies.ipynb (see efficiencies_nohup.out)")
    step_done "efficiencies notebook"

    step_start "plot_many_variables"
    python -u src/plot_many_variables.py > many_plots_nohup.out 2>&1 || failed+=("plot_many_variables (see many_plots_nohup.out)")
    step_done "plot_many_variables"

    step_start "p-value summary notebook"
    python -u src/execute_notebook.py ipynb_notebooks/plot_all_bdt_vars.ipynb > plot_all_bdt_vars_nohup.out 2>&1 \
        || failed+=("plot_all_bdt_vars.ipynb (see plot_all_bdt_vars_nohup.out)")
    step_done "p-value summary notebook"
fi

echo "[$(date '+%F %T')] all steps done ($(( (SECONDS - start_all) / 60 )) min total)"
if [ ${#failed[@]} -gt 0 ]; then
    printf 'FAILED: %s\n' "${failed[@]}"
    exit 1
fi
