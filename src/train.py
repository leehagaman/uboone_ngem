import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import argparse
import time

from signal_categories import topological_category_labels, train_category_labels
from ntuple_variables.variables import wc_training_vars, combined_training_vars, lantern_training_vars

from file_locations import intermediate_files_location

if __name__ == "__main__":
    main_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--training_vars", type=str, default="combined")
    parser.add_argument("--signal_categories", type=str, default="del1g_simple")
    parser.add_argument("--early_stopping_rounds", type=int, default=50,
                        help="Stop training if validation metric doesn't improve for this many rounds")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    if args.name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        args.name = f"training_{timestamp}"

    if args.training_vars == "combined":
        training_vars = combined_training_vars
    elif args.training_vars == "wc":
        training_vars = wc_training_vars
    elif args.training_vars == "lantern":
        training_vars = lantern_training_vars
    elif args.training_vars == "lantern_first_half":
        training_vars = lantern_training_vars[:len(lantern_training_vars)//2]
    else:
        raise ValueError(f"Invalid training_vars: {args.training_vars}")

    if args.signal_categories == "topological":
        signal_category_labels = topological_category_labels
        signal_category_var = "topological_signal_category"
    elif args.signal_categories == "del1g_simple":
        signal_category_labels = train_category_labels
        signal_category_var = "del1g_simple_signal_category"
    elif args.signal_categories == "nue_only":
        # Binary classification: 0 -> not_nue, 1 -> nue
        signal_category_labels = ["not_nue", "nue"]
        signal_category_var = "del1g_simple_signal_category"
    else:
        raise ValueError(f"Invalid signal_categories: {args.signal_categories}")

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

    all_df = pd.read_pickle(f"{intermediate_files_location}/presel_df_train_vars.pkl")

    # splitting into train and test, then re-making all_df
    no_data_df = all_df.query("filetype != 'data'").copy().reset_index()
    data_df = all_df.query("filetype == 'data'").copy().reset_index()
    train_indices, test_indices = train_test_split(np.arange(len(no_data_df)), test_size=0.5, random_state=42)
    data_df["used_for_training"] = False
    data_df["used_for_testing"] = False
    no_data_df["used_for_training"] = False
    no_data_df["used_for_testing"] = False
    no_data_df.loc[train_indices, "used_for_training"] = True
    no_data_df.loc[test_indices, "used_for_testing"] = True
    all_df = pd.concat([no_data_df, data_df])

    # Preselection: WC generic neutrino selection with at least one reco 20 MeV shower
    # (should already be applied in the presel_df_train_vars.pkl file)
    original_num_events = no_data_df.shape[0]
    presel_df = no_data_df.query("wc_kine_reco_Enu > 0")
    preselected_num_events = presel_df.shape[0]
    print(f"Preselected {preselected_num_events} / {original_num_events} events")

    x = presel_df[training_vars].to_numpy()
    w = presel_df["wc_net_weight"].to_numpy()

    num_categories = len(signal_category_labels)
    print(f"{num_categories=}")

    presel_train_df = presel_df.query("used_for_training == True")
    presel_test_df = presel_df.query("used_for_testing == True")

    x_train = presel_train_df[training_vars].to_numpy()
    x_train = x_train.astype(np.float64)
    x_train[(x_train > 1e10) | (x_train < -1e10)] = np.nan

    y_train = presel_train_df[signal_category_var].to_numpy()
    w_train = presel_train_df["wc_net_weight"].to_numpy()

    x_test = presel_test_df[training_vars].to_numpy()
    x_test = x_test.astype(np.float64)
    x_test[(x_test > 1e10) | (x_test < -1e10)] = np.nan

    y_test = presel_test_df[signal_category_var].to_numpy()
    w_test = presel_test_df["wc_net_weight"].to_numpy()

    num_training_vars = len(training_vars)
    print(f"{num_training_vars=}")
    
    # Debug: Check what categories are in training and test data
    #unique_categories_train = np.unique(y_train)
    #unique_categories_test = np.unique(y_test)
    #print(f"Categories in training data: {unique_categories_train}")
    #print(f"Categories in test data: {unique_categories_test}")
    #print(f"Expected categories: {list(range(num_categories))}")

    eval_set = [(x_train, y_train), (x_test, y_test)]
    eval_weights = [w_train, w_test]

    # Configure model and metrics for binary vs multi-class
    if num_categories == 2:
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=200,
            eval_metric=['logloss', 'error'],
            early_stopping_rounds=args.early_stopping_rounds,
        )
    else:
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_categories,
            n_estimators=200,
            eval_metric=['mlogloss', 'merror'],
            early_stopping_rounds=args.early_stopping_rounds,
        )

    model.fit(
        x_train, y_train, 
        sample_weight=w_train,
        eval_set=eval_set,
        sample_weight_eval_set=eval_weights,
        verbose=20
    )

    if model.best_iteration is not None:
        print(f"Early stopping: best_iteration={model.best_iteration}")

    print("Saving model...")
    model.get_booster().save_model(output_dir / "bdt.json")

    print("Creating feature importance plot...")
    plt.style.use('default')
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame({
        'feature': training_vars,
        'importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=True)  # Sort ascending so largest bar is at top
    top_20_df = importance_df.tail(20)
    plt.barh(range(len(top_20_df)), top_20_df['importance'])
    plt.yticks(range(len(top_20_df)), top_20_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance (Top 20)')
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Creating training curves...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    evals_result = model.evals_result()
    loss_key = 'mlogloss' if num_categories > 2 else 'logloss'
    err_key = 'merror' if num_categories > 2 else 'error'
    plt.plot(evals_result['validation_0'][loss_key], label='Train Loss', linewidth=2)
    plt.plot(evals_result['validation_1'][loss_key], label='Test Loss', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Multi-class Log Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(model.best_iteration, linestyle='--', color='k', alpha=0.6, label='Best iteration')
    plt.subplot(1, 2, 2)
    train_acc = [1 - err for err in evals_result['validation_0'][err_key]]
    test_acc = [1 - err for err in evals_result['validation_1'][err_key]]
    plt.plot(train_acc, label='Train Accuracy', linewidth=2)
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

    print("Creating probability histograms...")
    plt.figure(figsize=(20, 12))
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    n_categories = num_categories
    n_cols = 4
    n_rows = (n_categories + n_cols - 1) // n_cols
    bins = np.linspace(0, 1, 21)
    for i in range(n_categories):
        plt.subplot(n_rows, n_cols, i + 1)
        mask = (y_test == i)
        if np.sum(mask) > 0:
            plt.hist(y_proba[mask, i], bins=bins, histtype='step', label=f'True {train_category_labels[i]}', density=True)
            other_mask = (y_test != i)
            if np.sum(other_mask) > 0:
                plt.hist(y_proba[other_mask, i], bins=bins, histtype='step', label=f'Other categories', density=True)
        plt.xlabel(f'Probability for {train_category_labels[i]}')
        plt.ylabel('Density')
        plt.title(f'Probability Distribution: {train_category_labels[i]}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "probability_histograms.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Creating confusion matrix...")
    plt.figure(figsize=(20, 6))
    # Ensure confusion matrix includes all expected categories, even if they have zero events
    expected_labels = list(range(n_categories))
    cm = confusion_matrix(y_test, y_pred, sample_weight=w_test, labels=expected_labels)
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
    plt.xticks(range(n_categories), [train_category_labels[i] for i in range(n_categories)], rotation=45)
    plt.yticks(range(n_categories), [train_category_labels[i] for i in range(n_categories)])
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(cm_normalized_cols, cmap='Blues', aspect='auto')
    plt.colorbar(im2)
    plt.title('Confusion Matrix (Normalized Columns)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(n_categories):
        for j in range(n_categories):
            plt.text(j, i, f'{cm_normalized_cols[i, j]:.2f}', ha='center', va='center', fontsize=8)
    plt.xticks(range(n_categories), [train_category_labels[i] for i in range(n_categories)], rotation=45)
    plt.yticks(range(n_categories), [train_category_labels[i] for i in range(n_categories)])
    plt.subplot(1, 3, 3)
    im3 = plt.imshow(cm_normalized_rows, cmap='Blues', aspect='auto')
    plt.colorbar(im3)
    plt.title('Confusion Matrix (Normalized Rows)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(n_categories):
        for j in range(n_categories):
            plt.text(j, i, f'{cm_normalized_rows[i, j]:.2f}', ha='center', va='center', fontsize=8)
    plt.xticks(range(n_categories), [train_category_labels[i] for i in range(n_categories)], rotation=45)
    plt.yticks(range(n_categories), [train_category_labels[i] for i in range(n_categories)])
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Creating prediction dataframe...")
    # Get predictions for all data (not just test set, not just presel)
    x = all_df[training_vars].to_numpy()
    x = x.astype(np.float64)
    x[np.isinf(x)] = np.nan
    all_probabilities = model.predict_proba(x)
    prediction_df = pd.DataFrame()
    prediction_df['filetype'] = all_df['filetype']
    prediction_df['run'] = all_df['run']
    prediction_df['subrun'] = all_df['subrun']
    prediction_df['event'] = all_df['event']
    prediction_df['used_for_training'] = all_df['used_for_training']
    prediction_df['used_for_testing'] = all_df['used_for_testing']
    for i in range(n_categories):
        prediction_df[f'prob_{train_category_labels[i]}'] = all_probabilities[:, i]

    print("Saving predictions...")
    prediction_df.to_pickle(output_dir / "predictions.pkl")
    print(f"Saved predictions to: {output_dir / 'predictions.pkl'}")

    main_end_time = time.time()
    print(f"Total time to train and analyze the BDT: {main_end_time - main_start_time:.2f} seconds")
