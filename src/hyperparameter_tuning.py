import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import polars as pl
import xgboost as xgb
import optuna
from optuna.integration import XGBoostPruningCallback

from src.signal_categories import topological_category_labels, train_category_labels
from src.ntuple_variables.variables import (
    wc_training_vars, combined_training_vars, lantern_training_vars,
    glee_training_vars, pandora_training_vars, pandora_scalar_training_vars,
    combined_postprocessing_training_vars,
)
from src.file_locations import intermediate_files_location


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_vars", type=str, default="combined")
    parser.add_argument("--signal_categories", type=str, default="del1g_simple")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--db_path", type=str, default=None)
    return parser.parse_args()


def load_data(args):
    print("Loading dataframe...")
    all_df = pl.read_parquet(f"{intermediate_files_location}/presel_df_train_vars.parquet")

    no_data_df = all_df.filter(pl.col("filetype") != "data")

    # 40% train | 10% validation | 50% test (test not used here)
    no_data_df = no_data_df.with_columns([
        (pl.col("train_test_score") < 0.4).alias("used_for_training"),
        ((pl.col("train_test_score") >= 0.4) & (pl.col("train_test_score") < 0.5)).alias("used_for_validation"),
        (pl.col("train_test_score") >= 0.5).alias("used_for_testing"),
    ])

    if args.signal_categories == "nc_coh_1g_vs_bkg":
        signal_category_var = "filetype"
        signal_category_labels = ["not_NC_coherent_1g_reweighted", "NC_coherent_1g_reweighted"]
        presel_df = no_data_df.filter(pl.col("wc_kine_reco_Enu") > 0)
        presel_df = presel_df.filter(
            (pl.col("filetype") != "isotropic_one_gamma_overlay") &
            (pl.col("filetype") != "delete_one_gamma_overlay")
        )
        presel_df = presel_df.filter(
            ((pl.col("filetype") == "NC_coherent_1g_reweighted") & (pl.col("coherent_1g_keep") == True))
            | (pl.col("filetype") != "NC_coherent_1g_reweighted")
        )
    else:
        presel_df = no_data_df.filter(pl.col("wc_kine_reco_Enu") > 0)
        presel_df = presel_df.filter(
            (pl.col("filetype") != "numuCC_rad_corrected") &
            (pl.col("filetype") != "NC_coherent_1g_reweighted")
        )
        if args.signal_categories == "topological":
            signal_category_labels = topological_category_labels
            signal_category_var = "topological_signal_category"
        elif args.signal_categories == "del1g_simple":
            signal_category_labels = train_category_labels
            signal_category_var = "del1g_simple_signal_category"
        elif args.signal_categories == "nue_only":
            signal_category_labels = ["not_nue", "nue"]
            signal_category_var = "del1g_simple_signal_category"
        else:
            raise ValueError(f"Invalid signal_categories: {args.signal_categories}")

    train_df = presel_df.filter(pl.col("used_for_training"))
    val_df = presel_df.filter(pl.col("used_for_validation"))
    print(f"Train: {train_df.height}, Val: {val_df.height}")

    return train_df, val_df, signal_category_labels, signal_category_var


def build_matrices(train_df, val_df, training_vars, signal_category_var, is_nc_coh):
    def to_x(df):
        x = df.select(training_vars).to_numpy().astype(np.float64)
        x[(x > 1e10) | (x < -1e10)] = np.nan
        return x

    x_train = to_x(train_df)
    x_val = to_x(val_df)

    if is_nc_coh:
        w_train = np.ones(len(train_df))
        w_val = np.ones(len(val_df))
        y_train = train_df.select(
            (pl.col("filetype") == "NC_coherent_1g_reweighted") & pl.col("wc_truth_inFV")
        ).to_numpy().flatten().astype(int)
        y_val = val_df.select(
            (pl.col("filetype") == "NC_coherent_1g_reweighted") & pl.col("wc_truth_inFV")
        ).to_numpy().flatten().astype(int)
    else:
        w_train = train_df.select("wc_net_weight").to_numpy().flatten()
        w_val = val_df.select("wc_net_weight").to_numpy().flatten()
        y_train = train_df.select(signal_category_var).to_numpy().flatten()
        y_val = val_df.select(signal_category_var).to_numpy().flatten()

    dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(x_val, label=y_val, weight=w_val)
    return dtrain, dval


def make_objective(dtrain, dval, num_categories):
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
        params = {
            **base_params,
            "verbosity": 0,
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        }

        pruning_callback = XGBoostPruningCallback(trial, pruning_key)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dval, "validation")],
            early_stopping_rounds=30,
            callbacks=[pruning_callback],
            verbose_eval=False,
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

        return model.best_score

    return objective


if __name__ == "__main__":
    args = parse_args()

    if args.training_vars == "combined":
        training_vars = combined_training_vars
    elif args.training_vars == "wc":
        training_vars = wc_training_vars
    elif args.training_vars == "lantern":
        training_vars = lantern_training_vars
    elif args.training_vars == "lantern_first_half":
        training_vars = lantern_training_vars[:len(lantern_training_vars) // 2]
    elif args.training_vars == "glee":
        training_vars = glee_training_vars
    elif args.training_vars == "pandora":
        training_vars = pandora_training_vars
    elif args.training_vars == "pandora_scalars":
        training_vars = pandora_scalar_training_vars
    elif args.training_vars == "only_wc_lantern_combined":
        training_vars = wc_training_vars + lantern_training_vars + combined_postprocessing_training_vars
    else:
        raise ValueError(f"Invalid training_vars: {args.training_vars}")

    train_df, val_df, signal_category_labels, signal_category_var = load_data(args)
    num_categories = len(signal_category_labels)
    is_nc_coh = (args.signal_categories == "nc_coh_1g_vs_bkg")

    dtrain, dval = build_matrices(train_df, val_df, training_vars, signal_category_var, is_nc_coh)

    study_name = args.study_name or f"hparam_{args.signal_categories}_{args.training_vars}"
    db_path = args.db_path or f"optuna_{study_name}.db"

    # MedianPruner: prune if intermediate loss is above the median of completed
    # trials at the same boosting round. n_warmup_steps=20 avoids pruning too early.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)

    study = optuna.create_study(
        storage=f"sqlite:///{db_path}",
        study_name=study_name,
        direction="minimize",   # minimizing logloss / mlogloss
        pruner=pruner,
        load_if_exists=True,
    )

    # Enqueue a trial with XGBoost defaults (from train.py) as a baseline,
    # but only if it hasn't been run yet (safe to re-run on study resume).
    # reg_alpha default is 0.0 but our range is log-scale, so use 1e-4.
    default_params = {
        "max_depth": 6,
        "learning_rate": 0.3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "reg_alpha": 1e-3,
    }
    if default_params not in [t.params for t in study.trials]:
        print("adding trial to queue with default parameters...")
        study.enqueue_trial(default_params)

    objective = make_objective(dtrain, dval, num_categories)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\nTrials — complete: {complete}, pruned: {pruned}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best log-loss: {study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

# View the study by running:
# optuna-dashboard sqlite:///optuna_first_test_xgb_hp.db
