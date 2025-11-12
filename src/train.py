import numpy as np
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
import time

from signal_categories import topological_category_labels, train_category_labels
from ntuple_variables.variables import wc_training_vars, combined_training_vars, lantern_training_vars, glee_training_vars, pandora_training_vars, pandora_scalar_training_vars, combined_postprocessing_training_vars

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

    all_df = pl.read_parquet(f"{intermediate_files_location}/presel_df_train_vars.parquet")

    # splitting into train and test, then re-making all_df
    no_data_df = all_df.filter(pl.col("filetype") != "data")
    data_df = all_df.filter(pl.col("filetype") == "data")
    train_indices, test_indices = train_test_split(np.arange(no_data_df.height), test_size=0.5, random_state=42)
    
    # Create boolean arrays for train/test flags
    used_for_training = np.zeros(no_data_df.height, dtype=bool)
    used_for_testing = np.zeros(no_data_df.height, dtype=bool)
    used_for_training[train_indices] = True
    used_for_testing[test_indices] = True
    
    no_data_df = no_data_df.with_columns([
        pl.Series("used_for_training", used_for_training),
        pl.Series("used_for_testing", used_for_testing)
    ])
    
    data_df = data_df.with_columns([
        pl.lit(False).alias("used_for_training"),
        pl.lit(False).alias("used_for_testing")
    ])
    
    all_df = pl.concat([no_data_df, data_df])

    # Preselection: WC generic neutrino selection
    # (should already be applied in the presel_df_train_vars.pkl file)
    original_num_events = no_data_df.height
    presel_df = no_data_df.filter(pl.col("wc_kine_reco_Enu") > 0)
    preselected_num_events = presel_df.height
    print(f"Preselected {preselected_num_events} / {original_num_events} events")

    num_categories = len(signal_category_labels)
    print(f"{num_categories=}")

    presel_train_df = presel_df.filter(pl.col("used_for_training") == True)
    presel_test_df = presel_df.filter(pl.col("used_for_testing") == True)
    del presel_df

    # check for duplicates in training_vars, and print them
    training_vars_set = set(training_vars)
    if len(training_vars_set) != len(training_vars):
        print("Duplicates in training_vars:")
        for var in training_vars:
            if training_vars.count(var) > 1:
                print(var)
        raise ValueError("Duplicates in training_vars!")

    x_train = presel_train_df.select(training_vars).to_numpy()
    x_train = x_train.astype(np.float64)
    x_train[(x_train > 1e10) | (x_train < -1e10)] = np.nan

    y_train = presel_train_df.select(signal_category_var).to_numpy()
    y_train = y_train.flatten() if y_train.ndim > 1 else y_train
    w_train = presel_train_df.select("wc_net_weight").to_numpy()
    w_train = w_train.flatten() if w_train.ndim > 1 else w_train

    x_test = presel_test_df.select(training_vars).to_numpy()
    x_test = x_test.astype(np.float64)
    x_test[(x_test > 1e10) | (x_test < -1e10)] = np.nan

    y_test = presel_test_df.select(signal_category_var).to_numpy()
    y_test = y_test.flatten() if y_test.ndim > 1 else y_test
    w_test = presel_test_df.select("wc_net_weight").to_numpy()
    w_test = w_test.flatten() if w_test.ndim > 1 else w_test

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
        verbose=10
    )

    if model.best_iteration is not None:
        print(f"Early stopping: best_iteration={model.best_iteration}")

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
    loss_key = 'mlogloss' if num_categories > 2 else 'logloss'
    err_key = 'merror' if num_categories > 2 else 'error'
    train_loss = evals_result['validation_0'][loss_key]
    test_loss = evals_result['validation_1'][loss_key]
    plt.plot(train_loss, label='Train Loss', linewidth=2)
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
    
    # Save training curves to CSV
    print("Saving training curves to CSV...")
    num_iterations = len(train_loss)
    training_curves_df = pl.DataFrame({
        'iteration': list(range(num_iterations)),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'is_best_iteration': [i == model.best_iteration for i in range(num_iterations)]
    })
    training_curves_df.write_csv(output_dir / "training_curves.csv")
    print(f"Saved training curves to: {output_dir / 'training_curves.csv'}")

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
    x = all_df.select(training_vars).to_numpy()
    x = x.astype(np.float64)
    x[np.isinf(x)] = np.nan
    all_probabilities = model.predict_proba(x)
    
    # Build prediction dataframe columns
    prediction_cols = [
        all_df.select("filetype"),
        all_df.select("run"),
        all_df.select("subrun"),
        all_df.select("event"),
        all_df.select("used_for_training"),
        all_df.select("used_for_testing")
    ]
    for i in range(n_categories):
        prediction_cols.append(pl.DataFrame({
            f'prob_{train_category_labels[i]}': all_probabilities[:, i]
        }))
    
    prediction_df = pl.concat(prediction_cols, how="horizontal")

    print("Saving predictions...")
    prediction_df.write_parquet(output_dir / "predictions.parquet")
    print(f"Saved predictions to: {output_dir / 'predictions.parquet'}")

    main_end_time = time.time()
    print(f"Total time to train and analyze the BDT: {main_end_time - main_start_time:.2f} seconds")
