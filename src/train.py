import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import argparse

from signal_categories import topological_category_labels
from variables import wc_training_vars


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    if args.name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        args.name = f"training_{timestamp}"

    
    # Delete the directory if it exists
    if (PROJECT_ROOT / 'training_outputs' / args.name).exists():
        import os
        os.system(f"rm -rf {PROJECT_ROOT / 'training_outputs' / args.name}")
        print(f"Deleted existing directory: {PROJECT_ROOT / 'training_outputs' / args.name}")

    output_dir = PROJECT_ROOT / 'training_outputs' / args.name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")


    print("xgboost version: ", xgb.__version__)

    print("loading dataframe...")

    with open(f"{PROJECT_ROOT}/intermediate_files/generic_df_train_vars.pkl", "rb") as f:
        all_df = pickle.load(f)

    train_indices, test_indices = train_test_split(np.arange(len(all_df)), test_size=0.5, random_state=42)
    all_df["used_for_training"] = False
    all_df["used_for_testing"] = False
    all_df.loc[train_indices, "used_for_training"] = True
    all_df.loc[test_indices, "used_for_testing"] = True

    # Preselection: WC generic neutrino selection with at least one reco 20 MeV shower
    original_num_events = all_df.shape[0]
    presel_df = all_df.query("wc_kine_reco_Enu > 0 and wc_shw_sp_n_20mev_showers > 0") # the generic selection wc_kine_reco_Enu > 0 was already applied
    preselected_num_events = presel_df.shape[0]
    print(f"Preselected {preselected_num_events} / {original_num_events} events")

    x = presel_df[wc_training_vars].to_numpy()
    w = presel_df["wc_net_weight"].to_numpy()

    num_categories = len(topological_category_labels)
    print(f"{num_categories=}")

    presel_train_df = presel_df.query("used_for_training == True")
    presel_test_df = presel_df.query("used_for_testing == True")

    topological_signal_category_mapping = {cat: i for i, cat in enumerate(topological_category_labels)}

    x_train = presel_train_df[wc_training_vars].to_numpy()
    y_train = presel_train_df["topological_signal_category"].map(topological_signal_category_mapping).to_numpy()
    w_train = presel_train_df["wc_net_weight"].to_numpy()
    x_test = presel_test_df[wc_training_vars].to_numpy()
    y_test = presel_test_df["topological_signal_category"].map(topological_signal_category_mapping).to_numpy()
    w_test = presel_test_df["wc_net_weight"].to_numpy()
    
    # Debug: Check what categories are in training and test data
    #unique_categories_train = np.unique(y_train)
    #unique_categories_test = np.unique(y_test)
    #print(f"Categories in training data: {unique_categories_train}")
    #print(f"Categories in test data: {unique_categories_test}")
    #print(f"Expected categories: {list(range(num_categories))}")

    eval_set = [(x_train, y_train), (x_test, y_test)]
    eval_weights = [w_train, w_test]

    model = xgb.XGBClassifier(
        num_class=num_categories,
        n_estimators=200,
        eval_metric=['mlogloss', 'merror'],
    )

    model.fit(
        x_train, y_train, 
        sample_weight=w_train,
        eval_set=eval_set,
        verbose=20
    )

    # Save model
    model.get_booster().save_model(output_dir / "bdt.json")

    plt.style.use('default')

    # Feature Importance Plot
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame({
        'feature': wc_training_vars,
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

    # Training Curves
    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(model.evals_result()['validation_0']['mlogloss'], label='Train Loss', linewidth=2)
    plt.plot(model.evals_result()['validation_1']['mlogloss'], label='Test Loss', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Multi-class Log Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy curves (1 - error)
    plt.subplot(1, 2, 2)
    train_acc = [1 - err for err in model.evals_result()['validation_0']['merror']]
    test_acc = [1 - err for err in model.evals_result()['validation_1']['merror']]
    plt.plot(train_acc, label='Train Accuracy', linewidth=2)
    plt.plot(test_acc, label='Test Accuracy', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Probability Histograms for Different Categories
    plt.figure(figsize=(20, 12))

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    category_names = {v: k for k, v in topological_signal_category_mapping.items()}
    n_categories = len(topological_signal_category_mapping)
    
    print(f"Model predict_proba shape: {y_proba.shape}")
    print(f"Expected number of categories: {n_categories}")
    print(f"Model n_classes_: {model.n_classes_}")
    n_cols = 4
    n_rows = (n_categories + n_cols - 1) // n_cols
    bins = np.linspace(0, 1, 21)
    for i in range(n_categories):
        plt.subplot(n_rows, n_cols, i + 1)
        mask = (y_test == i)
        if np.sum(mask) > 0:
            plt.hist(y_proba[mask, i], bins=bins, histtype='step', label=f'True {category_names[i]}', density=True)
            other_mask = (y_test != i)
            if np.sum(other_mask) > 0:
                plt.hist(y_proba[other_mask, i], bins=bins, histtype='step', label=f'Other categories', density=True)
        plt.xlabel(f'Probability for {category_names[i]}')
        plt.ylabel('Density')
        plt.title(f'Probability Distribution: {category_names[i]}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "probability_histograms.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(20, 6))

    # Ensure confusion matrix includes all expected categories, even if they have zero events
    expected_labels = list(range(len(topological_signal_category_mapping)))
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

    # Add text annotations
    for i in range(n_categories):
        for j in range(n_categories):
            plt.text(j, i, str(int(cm[i, j])), ha='center', va='center', fontsize=8)

    plt.xticks(range(n_categories), [category_names[i] for i in range(n_categories)], rotation=45)
    plt.yticks(range(n_categories), [category_names[i] for i in range(n_categories)])

    # Plot confusion matrix (normalized columns)
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(cm_normalized_cols, cmap='Blues', aspect='auto')
    plt.colorbar(im2)
    plt.title('Confusion Matrix (Normalized Columns)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(n_categories):
        for j in range(n_categories):
            plt.text(j, i, f'{cm_normalized_cols[i, j]:.2f}', ha='center', va='center', fontsize=8)
    plt.xticks(range(n_categories), [category_names[i] for i in range(n_categories)], rotation=45)
    plt.yticks(range(n_categories), [category_names[i] for i in range(n_categories)])

    # Plot confusion matrix (normalized rows)
    plt.subplot(1, 3, 3)
    im3 = plt.imshow(cm_normalized_rows, cmap='Blues', aspect='auto')
    plt.colorbar(im3)
    plt.title('Confusion Matrix (Normalized Rows)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(n_categories):
        for j in range(n_categories):
            plt.text(j, i, f'{cm_normalized_rows[i, j]:.2f}', ha='center', va='center', fontsize=8)
    plt.xticks(range(n_categories), [category_names[i] for i in range(n_categories)], rotation=45)
    plt.yticks(range(n_categories), [category_names[i] for i in range(n_categories)])

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Creating prediction dataframe...")

    # Get predictions for all data (not just test set, not just presel)

    x = all_df[wc_training_vars].to_numpy()

    all_probabilities = model.predict_proba(x)

    prediction_df = pd.DataFrame()
    prediction_df['filetype'] = all_df['filetype']
    prediction_df['run'] = all_df['run']
    prediction_df['subrun'] = all_df['subrun']
    prediction_df['event'] = all_df['event']
    prediction_df['used_for_training'] = all_df['used_for_training']
    prediction_df['used_for_testing'] = all_df['used_for_testing']
    for i, category_name in enumerate(category_names.values()):
        prediction_df[f'prob_{category_name}'] = all_probabilities[:, i]
    prediction_df.to_pickle(output_dir / "predictions.pkl")
    print(f"Saved predictions to: {output_dir / 'predictions.pkl'}")
