import sys
from pathlib import Path

# Add project root to path to allow imports with src. prefix
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import polars as pl
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix
from datetime import datetime
import os
import argparse
import json
import time

from bdt_training_setup import (
    get_training_vars, get_signal_categories, add_bdt_role_columns,
    select_training_events, get_labels, get_physics_weights, get_training_weights,
    class_prior_correction, PhysicsWeightedLoglossMetric, bdt_feature_matrix,
)
from file_locations import intermediate_files_location
from df_helpers import format_duration

if __name__ == "__main__":
    main_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--training_vars", type=str, default="combined")
    parser.add_argument("--signal_categories", type=str, default="del1g_simple")
    parser.add_argument("--params_json", type=str, default=None,
                        help="Hyperparameters from hyperparameter_tuning.py (its <study_name>_best_params.json); "
                             "default: XGBoost defaults")
    parser.add_argument("--weighted", action="store_true",
                        help="Train with the physics event weights (weight mode \"normalized\"); default: unit "
                             "weights (weight mode \"unit\"), see bdt_training_setup.WEIGHT_MODES")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate, overriding the one from --params_json (e.g. for a learning rate scan "
                             "with fixed tuned hyperparameters)")
    parser.add_argument("--n_estimators", type=int, default=5000,
                        help="Upper limit on boosting rounds; training normally ends by early stopping first")
    parser.add_argument("--early_stopping_rounds", type=int, default=50,
                        help="Stop training if validation metric doesn't improve for this many rounds")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    if args.name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        args.name = f"training_{timestamp}"

    weight_mode = "normalized" if args.weighted else "unit"
    print(f"{weight_mode=}")

    xgb_params = {}
    if args.params_json is not None:
        with open(args.params_json) as f:
            tuned = json.load(f)
        xgb_params = tuned["xgb_params"]
        print(f"Using hyperparameters from {args.params_json}: {xgb_params}")
        setup = tuned.get("setup", {})
        for key, current in [("training_vars", args.training_vars), ("signal_categories", args.signal_categories)]:
            if setup.get(key) != current:
                print(f"WARNING: hyperparameters were tuned with {key}={setup.get(key)}, training with {current}")
        if tuned["weight_mode"] != weight_mode:
            print(f"WARNING: hyperparameters were tuned with weight_mode={tuned['weight_mode']}, training with {weight_mode}")
    if args.learning_rate is not None:
        xgb_params = {**xgb_params, "learning_rate": args.learning_rate}
        print(f"learning rate: {args.learning_rate}")

    training_vars = get_training_vars(args.training_vars)
    signal_category_labels, signal_category_var = get_signal_categories(args.signal_categories)

    # Delete the directory if it exists
    if (PROJECT_ROOT / 'training_outputs' / args.name).exists():
        import os
        os.system(f"rm -rf {PROJECT_ROOT / 'training_outputs' / args.name}")
        print(f"Deleted existing directory: {PROJECT_ROOT / 'training_outputs' / args.name}")

    output_dir = PROJECT_ROOT / 'training_outputs' / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")
    score_vis_dir = output_dir / "score_vis"
    score_vis_dir.mkdir(parents=True, exist_ok=True)

    print("xgboost version: ", xgb.__version__)

    print("loading dataframe...")

    all_df = pl.read_parquet(f"{intermediate_files_location}/presel_df_train_vars.parquet")

    # Assign each event's BDT role (train / validation / test / not_trained) in place (avoids duplicating
    # the full dataframe with pl.concat). The validation split is used for early stopping, so the test
    # split stays untouched by the training.
    all_df = add_bdt_role_columns(all_df)

    # Preselection: WC generic neutrino selection
    # (should already be applied in the presel_df_train_vars.pkl file)
    original_num_events = all_df.filter(pl.col("bdt_role") != "not_trained").height
    presel_df = select_training_events(all_df, args.signal_categories)

    preselected_num_events = presel_df.height
    print(f"Preselected {preselected_num_events} / {original_num_events} events")

    num_categories = len(signal_category_labels)
    print(f"{num_categories=}")

    presel_train_df = presel_df.filter(pl.col("bdt_role") == "train")
    presel_val_df = presel_df.filter(pl.col("bdt_role") == "validation")
    presel_test_df = presel_df.filter(pl.col("bdt_role") == "test")
    del presel_df
    print(f"Train: {presel_train_df.height}, Val: {presel_val_df.height}, Test: {presel_test_df.height}")

    x_train = bdt_feature_matrix(presel_train_df, training_vars)
    x_val = bdt_feature_matrix(presel_val_df, training_vars)
    x_test = bdt_feature_matrix(presel_test_df, training_vars)

    y_train = get_labels(presel_train_df, args.signal_categories)
    y_val = get_labels(presel_val_df, args.signal_categories)
    y_test = get_labels(presel_test_df, args.signal_categories)

    # physics weights (for the evaluation plots) and the weights given to XGBoost
    w_phys_train = get_physics_weights(presel_train_df, args.signal_categories)
    w_phys_test = get_physics_weights(presel_test_df, args.signal_categories)
    normalization = w_phys_train.mean()
    w_train = get_training_weights(w_phys_train, weight_mode, normalization)
    w_phys_val = get_physics_weights(presel_val_df, args.signal_categories)
    w_val = get_training_weights(w_phys_val, weight_mode, normalization)
    w_test = get_training_weights(w_phys_test, weight_mode, normalization)
    prior_correction = class_prior_correction(y_train, w_train, w_phys_train, num_categories)
    print(f"mean physics training weight: {normalization:.4g}")
    print(f"class prior correction factors (physics / training priors): {np.array2string(prior_correction, precision=3)}")
    del presel_train_df, presel_val_df, presel_test_df

    print("num training signal:", np.sum(y_train == 1))
    print("num training background:", np.sum(y_train == 0))
    print(f"{y_train[:20]=}")

    num_training_vars = len(training_vars)
    print(f"{num_training_vars=}")

    print("creating eval_set and eval_weights...")

    # XGBoost early-stops on the last eval_set entry (validation); test is only monitored
    eval_set = [(x_train, y_train), (x_test, y_test), (x_val, y_val)]
    eval_weights = [w_train, w_test, w_val]

    # Log loss with the physics weights for every eval set regardless of weight_mode, so these
    # curves are comparable between weight modes (the built-in metrics use eval_weights)
    physics_logloss = PhysicsWeightedLoglossMetric(
        [("train", y_train, w_phys_train), ("test", y_test, w_phys_test), ("validation", y_val, w_phys_val)],
        num_categories, prior_correction)

    # Early stop on the built-in validation log loss (with the training weights, as in the
    # hyperparameter tuning), rather than XGBoost's default of the last (custom) metric
    loss_key = 'mlogloss' if num_categories > 2 else 'logloss'
    early_stopping = xgb.callback.EarlyStopping(
        rounds=args.early_stopping_rounds, metric_name=loss_key, data_name="validation_2", save_best=False)

    print("creating model...")

    # Configure model and metrics for binary vs multi-class
    if num_categories == 2:
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
            n_estimators=args.n_estimators,
            eval_metric=['logloss', 'error', physics_logloss],
            callbacks=[early_stopping],
            **xgb_params,
        )
    else:
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_categories,
            tree_method='hist',
            n_estimators=args.n_estimators,
            eval_metric=['mlogloss', 'merror', physics_logloss],
            callbacks=[early_stopping],
            **xgb_params,
        )

    print("fitting model...")

    fit_start_time = time.time()
    model.fit(
        x_train, y_train, 
        sample_weight=w_train,
        eval_set=eval_set,
        sample_weight_eval_set=eval_weights,
        verbose=10
    )

    fit_minutes = (time.time() - fit_start_time) / 60
    if model.best_iteration is not None:
        print(f"Early stopping: best_iteration={model.best_iteration}")
    print(f"fit time: {fit_minutes:.1f} min")

    # x_train is no longer needed after fitting; eval_set/eval_weights also hold
    # references to it, so drop them too to actually free the memory.
    del eval_set, eval_weights, x_train, y_train, w_train, x_val, y_val, w_val

    print("Saving training config...")
    with open(output_dir / "training_config.json", "w") as f:
        json.dump({
            "training_vars": args.training_vars,
            "signal_categories": args.signal_categories,
            "weight_mode": weight_mode,
            "params_json": args.params_json,
            "xgb_params": xgb_params,
            "n_estimators": args.n_estimators,
            "early_stopping_rounds": args.early_stopping_rounds,
            "best_iteration": model.best_iteration,
            "fit_minutes": fit_minutes,
            # comparable between weight modes (see physics_weighted_loss_curves.png)
            "physics_weighted_logloss_at_best_iteration": {
                set_name: {kind: values[model.best_iteration] for kind, values in curves.items()}
                for set_name, curves in physics_logloss.history.items()},
            "mean_physics_training_weight": float(normalization),
            # multiply the predicted probabilities by these (and renormalize) to move the class
            # priors to the physics priors; all ones for weight_mode "normalized"
            "class_prior_correction": dict(zip(signal_category_labels, prior_correction.tolist())),
        }, f, indent=2)

    print("Saving model...")
    model.get_booster().save_model(output_dir / "bdt.json")

    print("Creating feature importance plots...")
    plt.style.use('default')
    
    # Get importance scores for weight, gain, and cover
    booster = model.get_booster()
    
    # Get importance dictionaries (feature name -> importance value)
    weight_importance = booster.get_score(importance_type='weight')
    gain_importance = booster.get_score(importance_type='gain')
    cover_importance = booster.get_score(importance_type='cover')
    
    # Convert to arrays aligned with training_vars
    # XGBoost uses f0, f1, f2, ... as feature names
    def get_importance_array(importance_dict, feature_names):
        importance_array = []
        for i, feature_name in enumerate(feature_names):
            feature_key = f'f{i}'
            importance_array.append(importance_dict.get(feature_key, 0.0))
        return np.array(importance_array)
    
    weight_array = get_importance_array(weight_importance, training_vars)
    gain_array = get_importance_array(gain_importance, training_vars)
    cover_array = get_importance_array(cover_importance, training_vars)
    
    # Save feature importances to CSV
    print("Saving feature importances to CSV...")
    importance_df = pl.DataFrame({
        'feature': training_vars,
        'weight_importance': weight_array,
        'gain_importance': gain_array,
        'cover_importance': cover_array
    })
    importance_df.write_csv(output_dir / "feature_importances.csv")
    print(f"Saved feature importances to: {output_dir / 'feature_importances.csv'}")
    
    # Create plots for each importance type
    for importance_type, importance_array, importance_name in [
        ('weight', weight_array, 'Weight'),
        ('gain', gain_array, 'Gain'),
        ('cover', cover_array, 'Cover')
    ]:
        plt.figure(figsize=(12, 8))
        importance_df = pl.DataFrame({
            'feature': training_vars,
            'importance': importance_array
        })
        importance_df = importance_df.sort('importance')  # Sort ascending so largest bar is at top
        top_20_df = importance_df.tail(20)
        plt.barh(range(len(top_20_df)), top_20_df['importance'].to_numpy())
        plt.yticks(range(len(top_20_df)), top_20_df['feature'].to_list())
        plt.xlabel(f'Feature Importance ({importance_name})')
        plt.title(f'XGBoost Feature Importance - {importance_name} (Top 20)')
        plt.tight_layout()
        plt.savefig(output_dir / f"feature_importance_{importance_type}.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("Creating training curves...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    evals_result = model.evals_result()
    err_key = 'merror' if num_categories > 2 else 'error'
    train_loss = evals_result['validation_0'][loss_key]
    test_loss = evals_result['validation_1'][loss_key]
    val_loss = evals_result['validation_2'][loss_key]
    plt.plot(train_loss, label='Train Loss', linewidth=2)
    plt.plot(val_loss, label='Validation Loss', linewidth=2)
    plt.plot(test_loss, label='Test Loss', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Multi-class Log Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(model.best_iteration, linestyle='--', color='k', alpha=0.6, label='Best iteration')
    plt.subplot(1, 2, 2)
    train_acc = [1 - err for err in evals_result['validation_0'][err_key]]
    test_acc = [1 - err for err in evals_result['validation_1'][err_key]]
    val_acc = [1 - err for err in evals_result['validation_2'][err_key]]
    plt.plot(train_acc, label='Train Accuracy', linewidth=2)
    plt.plot(val_acc, label='Validation Accuracy', linewidth=2)
    plt.plot(test_acc, label='Test Accuracy', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(model.best_iteration, linestyle='--', color='k', alpha=0.6, label='Best iteration')
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training curves to CSV
    print("Saving training curves to CSV...")
    num_iterations = len(train_loss)
    training_curves_df = pl.DataFrame({
        'iteration': list(range(num_iterations)),
        'train_loss': train_loss,
        'validation_loss': val_loss,
        'test_loss': test_loss,
        'train_accuracy': train_acc,
        'validation_accuracy': val_acc,
        'test_accuracy': test_acc,
        'is_best_iteration': [i == model.best_iteration for i in range(num_iterations)]
    })
    training_curves_df.write_csv(output_dir / "training_curves.csv")
    print(f"Saved training curves to: {output_dir / 'training_curves.csv'}")

    # Physics-weighted log loss, comparable between weight modes. The prior-corrected version
    # first moves the class priors to the physics priors (identical for weight_mode "normalized").
    print("Creating physics-weighted loss curves...")
    plt.figure(figsize=(12, 5))
    for i_plot, (kind, kind_label) in enumerate([("raw", "Raw probabilities"), ("prior_corrected", "Prior-corrected probabilities")]):
        plt.subplot(1, 2, i_plot + 1)
        for set_name in ["train", "validation", "test"]:
            plt.plot(physics_logloss.history[set_name][kind], label=f'{set_name.capitalize()} Loss', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Physics-Weighted Log Loss')
        plt.title(f'{kind_label} (weight_mode={weight_mode})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axvline(model.best_iteration, linestyle='--', color='k', alpha=0.6, label='Best iteration')
    plt.tight_layout()
    plt.savefig(output_dir / "physics_weighted_loss_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    physics_curves_df = pl.DataFrame({
        'iteration': list(range(num_iterations)),
        **{f'{set_name}_physics_logloss{"" if kind == "raw" else "_prior_corrected"}': physics_logloss.history[set_name][kind]
           for set_name in ["train", "validation", "test"] for kind in ["raw", "prior_corrected"]},
        'is_best_iteration': [i == model.best_iteration for i in range(num_iterations)]
    })
    physics_curves_df.write_csv(output_dir / "physics_weighted_loss_curves.csv")
    print(f"Saved physics-weighted loss curves to: {output_dir / 'physics_weighted_loss_curves.csv'}")
    for set_name in ["validation", "test"]:
        print(f"{set_name} physics-weighted log loss at best iteration: "
              f"raw {physics_logloss.history[set_name]['raw'][model.best_iteration]:.6f}, "
              f"prior-corrected {physics_logloss.history[set_name]['prior_corrected'][model.best_iteration]:.6f}")

    print("Creating probability histograms...")
    plt.figure(figsize=(20, 12))
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    del x_test  # no longer needed; y_pred/y_proba/y_test/w_phys_test carry the rest
    n_categories = num_categories
    n_cols = 4
    n_rows = (n_categories + n_cols - 1) // n_cols
    bins = np.linspace(0, 1, 21)
    for i in range(n_categories):
        plt.subplot(n_rows, n_cols, i + 1)
        mask = (y_test == i)
        if np.sum(mask) > 0:
            plt.hist(y_proba[mask, i], bins=bins, histtype='step', label=f'True {signal_category_labels[i]}', density=True)
            other_mask = (y_test != i)
            if np.sum(other_mask) > 0:
                plt.hist(y_proba[other_mask, i], bins=bins, histtype='step', label=f'Other categories', density=True)
        plt.xlabel(f'Probability for {signal_category_labels[i]}')
        plt.ylabel('Density')
        plt.title(f'Probability Distribution: {signal_category_labels[i]}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "probability_histograms.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Creating confusion matrix...")
    plt.figure(figsize=(20, 6))
    # Ensure confusion matrix includes all expected categories, even if they have zero events
    expected_labels = list(range(n_categories))
    cm = confusion_matrix(y_test, y_pred, sample_weight=w_phys_test, labels=expected_labels)
    # Handle division by zero for normalization
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    # Normalize rows (avoid division by zero)
    cm_normalized_rows = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        if row_sums[i] > 0:
            cm_normalized_rows[i, :] = cm[i, :] / row_sums[i]
    # Normalize columns (avoid division by zero)
    cm_normalized_cols = np.zeros_like(cm, dtype=float)
    for j in range(cm.shape[1]):
        if col_sums[j] > 0:
            cm_normalized_cols[:, j] = cm[:, j] / col_sums[j]
    # Plot confusion matrix (counts)
    plt.subplot(1, 3, 1)
    im1 = plt.imshow(cm, cmap='Blues', aspect='auto', norm=LogNorm())
    plt.colorbar(im1)
    plt.title('Confusion Matrix (Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    print(f"{n_categories=}")
    print(f"{cm.shape=}")
    for i in range(n_categories):
        for j in range(n_categories):
            plt.text(j, i, str(int(cm[i, j])), ha='center', va='center', fontsize=8)
    plt.xticks(range(n_categories), [signal_category_labels[i] for i in range(n_categories)], rotation=45)
    plt.yticks(range(n_categories), [signal_category_labels[i] for i in range(n_categories)])
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(cm_normalized_cols, cmap='Blues', aspect='auto')
    plt.colorbar(im2)
    plt.title('Confusion Matrix (Normalized Columns)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(n_categories):
        for j in range(n_categories):
            plt.text(j, i, f'{cm_normalized_cols[i, j]:.2f}', ha='center', va='center', fontsize=8)
    plt.xticks(range(n_categories), [signal_category_labels[i] for i in range(n_categories)], rotation=45)
    plt.yticks(range(n_categories), [signal_category_labels[i] for i in range(n_categories)])
    plt.subplot(1, 3, 3)
    im3 = plt.imshow(cm_normalized_rows, cmap='Blues', aspect='auto')
    plt.colorbar(im3)
    plt.title('Confusion Matrix (Normalized Rows)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(n_categories):
        for j in range(n_categories):
            plt.text(j, i, f'{cm_normalized_rows[i, j]:.2f}', ha='center', va='center', fontsize=8)
    plt.xticks(range(n_categories), [signal_category_labels[i] for i in range(n_categories)], rotation=45)
    plt.yticks(range(n_categories), [signal_category_labels[i] for i in range(n_categories)])
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Creating prediction dataframe...")
    # Get predictions for all data (not just test set, not just presel).
    # Predict in chunks to bound peak memory: the full matrix would be ~3.8M x 1000 floats.
    # bdt_feature_matrix applies the same missing-value preprocessing as for the training.
    prediction_chunk_size = 200_000
    n_prediction_rows = all_df.height
    prob_chunks = []
    for start in range(0, n_prediction_rows, prediction_chunk_size):
        print(f"predicting chunk starting at {start}...")
        x_chunk = bdt_feature_matrix(all_df.slice(start, prediction_chunk_size), training_vars)
        prob_chunks.append(model.predict_proba(x_chunk))
        del x_chunk
    print("concatenating chunks...")
    all_probabilities = np.concatenate(prob_chunks, axis=0)
    del prob_chunks

    # Build prediction dataframe columns
    print("making prediction_cols...")
    prediction_cols = [
        all_df.select("filename"),
        all_df.select("filetype"),
        all_df.select("run"),
        all_df.select("subrun"),
        all_df.select("event"),
        # how each event may be used downstream (df_helpers.select_bdt_test_events)
        all_df.select("bdt_role"),
        all_df.select("bdt_prediction_weight_factor"),
    ]
    for i in range(n_categories):
        print(f"adding prediction column {i}...")
        prediction_cols.append(pl.DataFrame({
            f'prob_{signal_category_labels[i]}': all_probabilities[:, i]
        }))
    
    print("making prediction_df...")
    prediction_df = pl.concat(prediction_cols, how="horizontal")

    print("checking for duplicates...")
    dup_mask = pl.struct("filename", "filetype", "run", "subrun", "event").is_duplicated()
    n_dups = prediction_df.select(dup_mask.sum()).item()
    if n_dups > 0:
        dups = prediction_df.filter(dup_mask).select(["filename", "filetype", "run", "subrun", "event"])
        print(f"Found {n_dups} duplicate rows in predictions, first 10:\n{dups.head(10)}")
        raise ValueError("Duplicate filename/filetype/run/subrun/event in predictions!")

    print("Saving predictions...")
    prediction_df.write_parquet(output_dir / "predictions.parquet")
    print(f"Saved predictions to: {output_dir / 'predictions.parquet'}")

    main_end_time = time.time()
    print(f"Total time to train and analyze the BDT: {format_duration(main_end_time - main_start_time)}")
