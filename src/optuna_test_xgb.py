"""
Optuna hyperparameter optimization for XGBoost with pruning.
Pruning cuts unpromising trials early based on intermediate AUC scores.
"""

import optuna
from optuna.integration import XGBoostPruningCallback
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Generate toy classification data
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42,
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)


def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "verbosity": 0,
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    }

    # XGBoostPruningCallback reports validation AUC to Optuna after each round,
    # allowing Median pruner to kill unpromising trials early.
    pruning_callback = XGBoostPruningCallback(trial, "validation-auc")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "validation")],
        early_stopping_rounds=30,
        callbacks=[pruning_callback],
        verbose_eval=False,
    )

    # Raise TrialPruned if the callback decided this trial should stop
    if trial.should_prune():
        raise optuna.TrialPruned()

    preds = model.predict(dval)
    return roc_auc_score(y_val, preds)


if __name__ == "__main__":
    # MedianPruner: prune if a trial's intermediate value is below the median
    # of completed trials at the same step. startup_trials=5 means the first
    # 5 trials always run to completion to establish a baseline.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)

    study = optuna.create_study(
        storage="sqlite:///optuna_test_xgb.db",
        study_name="xgb_toy",
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\nTrials — complete: {complete}, pruned: {pruned}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best AUC:   {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

# View the study by running:
# optuna-dashboard sqlite:///optuna_test_xgb.db

