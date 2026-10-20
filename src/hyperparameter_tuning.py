import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import hashlib
import json
import os
from datetime import datetime

import numpy as np
import polars as pl
import xgboost as xgb
import optuna
from optuna.integration import XGBoostPruningCallback

from bdt_training_setup import (
    WEIGHT_MODES, get_training_vars, get_signal_categories, add_split_columns,
    select_training_events, get_labels, get_physics_weights, get_training_weights,
    class_prior_correction, apply_prior_correction, weighted_logloss, bdt_feature_matrix,
)
from file_locations import intermediate_files_location

INPUT_FILE = f"{intermediate_files_location}/presel_df_train_vars.parquet"

# XGBoost defaults, enqueued as the first trial of a new study as a baseline.
# reg_alpha default is 0.0 but our range is log-scale, so use the bottom of the range.
DEFAULT_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.3,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 1e-4,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_vars", type=str, default="combined")
    parser.add_argument("--signal_categories", type=str, default="del1g_simple")
    parser.add_argument("--weight_mode", type=str, default="normalized", choices=WEIGHT_MODES,
                        help="Event weights used in the XGBoost training (see bdt_training_setup.WEIGHT_MODES). "
                             "Run each weight mode as its own study, and compare their best trials with the "
                             "val_physics_weighted_logloss trial attribute.")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--study_name", type=str, default=None,
                        help="Default: a new timestamped study. Pass an existing name to resume it.")
    parser.add_argument("--db_path", type=str, default=None,
                        help="Default: <project>/hyperparameter_tuning_outputs/<study_name>.db")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Fixed learning rate for all trials (so pruning compares trials at the same boosting step fairly)")
    parser.add_argument("--tune_learning_rate", action="store_true",
                        help="Also tune the learning rate (log-uniform in [0.01, 0.3]); disables pruning, since "
                             "slow-learning trials would be pruned for being slow rather than worse")
    parser.add_argument("--no_pruning", action="store_true")
    parser.add_argument("--max_boost_rounds", type=int, default=5000,
                        help="Upper limit on boosting rounds; trials normally end by early stopping first")
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--nthread", type=int, default=0, help="XGBoost threads (0 = all cores)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optuna sampler seed (use different seeds for parallel processes sharing one study)")
    parser.add_argument("--allow_setup_mismatch", action="store_true",
                        help="Allow resuming a study whose input file / variables / settings differ from this run")
    return parser.parse_args()


def load_data(args, training_vars):
    print("Loading dataframe...")
    all_df = pl.read_parquet(INPUT_FILE)
    presel_df = select_training_events(add_split_columns(all_df), args.signal_categories)
    del all_df

    train_df = presel_df.filter(pl.col("used_for_training"))
    val_df = presel_df.filter(pl.col("used_for_validation"))
    del presel_df
    print(f"Train: {train_df.height}, Val: {val_df.height}")

    data = {}
    for split, df in [("train", train_df), ("val", val_df)]:
        data[f"x_{split}"] = bdt_feature_matrix(df, training_vars)
        data[f"y_{split}"] = get_labels(df, args.signal_categories)
        data[f"w_phys_{split}"] = get_physics_weights(df, args.signal_categories)
    return data


def make_objective(args, dtrain, dval, num_categories, y_val, w_phys_val, prior_correction):
    is_binary = (num_categories == 2)
    if is_binary:
        base_params = {"objective": "binary:logistic", "eval_metric": "logloss"}
        pruning_key = "validation-logloss"
    else:
        base_params = {
            "objective": "multi:softprob",
            "num_class": num_categories,
            "eval_metric": "mlogloss",
        }
        pruning_key = "validation-mlogloss"

    def objective(trial):
        if args.tune_learning_rate:
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        else:
            learning_rate = args.learning_rate

        # min_child_weight and reg_lambda are sums of hessian*weight; with weights of order one
        # (both weight modes), the per-event hessian is <= 0.25, so these are in units of ~events.
        params = {
            **base_params,
            "tree_method": "hist",
            "nthread": args.nthread,
            "verbosity": 0,
            "learning_rate": learning_rate,
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 1000.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        }

        callbacks = []
        if not (args.no_pruning or args.tune_learning_rate):
            callbacks.append(XGBoostPruningCallback(trial, pruning_key))

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=args.max_boost_rounds,
            evals=[(dval, "validation")],
            early_stopping_rounds=args.early_stopping_rounds,
            callbacks=callbacks,
            verbose_eval=False,
        )

        # Comparable across weight modes: physics-weighted validation log loss, of the raw
        # probabilities and after moving the class priors to the physics priors (the same for
        # the "normalized" weight mode)
        probs = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        if is_binary:
            probs = np.column_stack([1 - probs, probs])
        trial.set_user_attr("val_physics_weighted_logloss_raw", weighted_logloss(probs, y_val, w_phys_val))
        probs = apply_prior_correction(probs, prior_correction)
        trial.set_user_attr("val_physics_weighted_logloss", weighted_logloss(probs, y_val, w_phys_val))
        trial.set_user_attr("best_iteration", model.best_iteration)
        trial.set_user_attr("learning_rate", learning_rate)

        return model.best_score

    return objective


def setup_fingerprint(args, training_vars, data):
    """Everything that makes trials comparable; a resumed study must match it."""
    return {
        "input_file": INPUT_FILE,
        "input_file_mtime": os.path.getmtime(INPUT_FILE),
        "training_vars": args.training_vars,
        "training_vars_sha1": hashlib.sha1("\n".join(training_vars).encode()).hexdigest(),
        "num_training_vars": len(training_vars),
        "signal_categories": args.signal_categories,
        "weight_mode": args.weight_mode,
        "n_train": len(data["y_train"]),
        "n_val": len(data["y_val"]),
        "learning_rate": "tuned" if args.tune_learning_rate else args.learning_rate,
        "max_boost_rounds": args.max_boost_rounds,
        "early_stopping_rounds": args.early_stopping_rounds,
    }


if __name__ == "__main__":
    args = parse_args()

    training_vars = get_training_vars(args.training_vars)
    signal_category_labels, _ = get_signal_categories(args.signal_categories)
    num_categories = len(signal_category_labels)

    data = load_data(args, training_vars)

    # one normalization (the mean physics training weight) for every split
    normalization = data["w_phys_train"].mean()
    w_train = get_training_weights(data["w_phys_train"], args.weight_mode, normalization)
    w_val = get_training_weights(data["w_phys_val"], args.weight_mode, normalization)
    prior_correction = class_prior_correction(data["y_train"], w_train, data["w_phys_train"], num_categories)
    print(f"weight_mode={args.weight_mode}, mean physics training weight={normalization:.4g}")
    print(f"class prior correction factors: {np.array2string(prior_correction, precision=3)}")

    fingerprint = setup_fingerprint(args, training_vars, data)

    print("Building QuantileDMatrices...")
    dtrain = xgb.QuantileDMatrix(data["x_train"], label=data["y_train"], weight=w_train, nthread=args.nthread)
    dval = xgb.QuantileDMatrix(data["x_val"], label=data["y_val"], weight=w_val, ref=dtrain, nthread=args.nthread)
    del data["x_train"], data["x_val"]

    if args.study_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.study_name = f"hparam_{args.signal_categories}_{args.training_vars}_{args.weight_mode}_{timestamp}"
    output_dir = project_root / "hyperparameter_tuning_outputs"
    output_dir.mkdir(exist_ok=True)
    db_path = Path(args.db_path) if args.db_path else output_dir / f"{args.study_name}.db"

    # MedianPruner: prune if the validation loss is above the median of completed trials at the
    # same boosting round. Only fair with a fixed learning rate.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)

    study = optuna.create_study(
        storage=f"sqlite:///{db_path.resolve()}",
        study_name=args.study_name,
        direction="minimize",   # minimizing logloss / mlogloss
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )

    stored_fingerprint = study.user_attrs.get("setup")
    if stored_fingerprint is None:
        if len(study.trials) > 0 and not args.allow_setup_mismatch:
            raise ValueError(f"Study '{args.study_name}' already has trials but no recorded setup "
                             "(made by an older version of this script); use a new --study_name")
        study.set_user_attr("setup", fingerprint)
    elif stored_fingerprint != fingerprint:
        diffs = {k: (stored_fingerprint.get(k), fingerprint.get(k))
                 for k in set(stored_fingerprint) | set(fingerprint)
                 if stored_fingerprint.get(k) != fingerprint.get(k)}
        message = f"Resuming study '{args.study_name}' with a different setup (stored, current): {diffs}"
        if not args.allow_setup_mismatch:
            raise ValueError(message + "\nUse a new --study_name, or pass --allow_setup_mismatch")
        print("WARNING: " + message)

    # Enqueue the XGBoost defaults as a baseline for a new study
    if len(study.trials) == 0:
        print("adding trial to queue with default parameters...")
        default_params = dict(DEFAULT_PARAMS)
        if not args.tune_learning_rate:
            default_params.pop("learning_rate")
        study.enqueue_trial(default_params)

    objective = make_objective(args, dtrain, dval, num_categories,
                               data["y_val"], data["w_phys_val"], prior_correction)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    best = study.best_trial
    print(f"\nTrials — complete: {complete}, pruned: {pruned}")
    print(f"Best trial: {best.number}")
    print(f"Best validation log-loss ({args.weight_mode} weights): {best.value:.6f}")
    print(f"  physics-weighted, raw: {best.user_attrs['val_physics_weighted_logloss_raw']:.6f}")
    print(f"  physics-weighted, prior-corrected: {best.user_attrs['val_physics_weighted_logloss']:.6f}")
    print(f"  best_iteration: {best.user_attrs['best_iteration']}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # train.py --params_json reads this
    best_params = {
        "study_name": args.study_name,
        "trial_number": best.number,
        "weight_mode": args.weight_mode,
        "xgb_params": {**best.params, "learning_rate": best.user_attrs["learning_rate"]},
        "best_iteration": best.user_attrs["best_iteration"],
        "val_logloss": best.value,
        "val_physics_weighted_logloss_raw": best.user_attrs["val_physics_weighted_logloss_raw"],
        "val_physics_weighted_logloss": best.user_attrs["val_physics_weighted_logloss"],
        "setup": fingerprint,
    }
    best_params_path = output_dir / f"{args.study_name}_best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best params to: {best_params_path}")
    print(f"View the study with: optuna-dashboard sqlite:///{db_path.resolve()}")
