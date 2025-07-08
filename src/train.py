#%%

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from variables import wc_training_vars

print("xgboost version: ", xgb.__version__)

print("loading dataframe...")

with open(f"{PROJECT_ROOT}/intermediate_files/all_df.pkl", "rb") as f:
    all_df = pickle.load(f)

#%%

x = all_df[wc_training_vars].to_numpy()
w = all_df["wc_net_weight"].to_numpy()

physics_signal_category_mapping = {
    "NCDeltaRad_1gNp": 0,
    "NCDeltaRad_1g0p": 1,
    "NC1pi0_Np": 2,
    "NC1pi0_0p": 3,
    "numuCC1pi0_Np": 4,
    "numuCC1pi0_0p": 5,
    "pi0_outFV": 6,
    "other": 7,
}

reconstructable_signal_category_mapping = {
    "1gNp": 0,
    "1g0p": 1,
    "1gNp1mu": 2,
    "1g0p1mu": 3,
    "2gNp": 4,
    "2g0p": 5,
    "2gNp1mu": 6,
    "2g0p1mu": 7,
    "1g_outFV": 8,
    "2g_outFV": 9,
    "other": 10,
}

y = all_df["reconstructable_signal_category"].map(reconstructable_signal_category_mapping).to_numpy()

# Split data for evaluation
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    x, y, w, test_size=0.2, random_state=42, stratify=y
)

#%%

# Create evaluation sets for monitoring training
eval_set = [(X_train, y_train), (X_test, y_test)]
eval_weights = [w_train, w_test]

model = xgb.XGBClassifier(
    num_class=11,
    n_estimators=200,
    eval_metric=['mlogloss', 'merror'],
    early_stopping_rounds=10,
)

# Fit with evaluation sets to capture training curves
model.fit(
    X_train, y_train, 
    sample_weight=w_train,
    eval_set=eval_set,
    verbose=20
)

#%%

# Save model
model.get_booster().save_model(f"{PROJECT_ROOT}/models/reconstructable_signal_category_bdt.json")

#%%

# Generate plots
plt.style.use('default')

# Create output directory for plots
plots_dir = PROJECT_ROOT / "plots"

# Feature Importance Plot
plt.figure(figsize=(12, 8))
importance_df = pd.DataFrame({
    'feature': wc_training_vars,
    'importance': model.feature_importances_
})
importance_df = importance_df.sort_values('importance', ascending=True)  # Sort ascending so largest bar is at top

# Keep only top 20 features
top_20_df = importance_df.tail(20)  # Use tail to get the 20 most important features

plt.barh(range(len(top_20_df)), top_20_df['importance'])
plt.yticks(range(len(top_20_df)), top_20_df['feature'])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance (Top 20)')
plt.tight_layout()
plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')


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
plt.savefig(plots_dir / "training_curves.png", dpi=300, bbox_inches='tight')


# Probability Histograms for Different Categories
plt.figure(figsize=(20, 12))

# Get predictions and probabilities
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Create reverse mapping for category names
category_names = {v: k for k, v in reconstructable_signal_category_mapping.items()}

# Plot probability histograms for each category
n_categories = len(reconstructable_signal_category_mapping)
n_cols = 4
n_rows = (n_categories + n_cols - 1) // n_cols

for i in range(n_categories):
    plt.subplot(n_rows, n_cols, i + 1)
    
    # Get true labels for this category
    mask = (y_test == i)
    
    if np.sum(mask) > 0:
        # Plot probability distribution for this category
        plt.hist(y_proba[mask, i], bins=50, alpha=0.7, 
                label=f'True {category_names[i]}', density=True)
        
        # Plot probability distribution for other categories
        other_mask = (y_test != i)
        if np.sum(other_mask) > 0:
            plt.hist(y_proba[other_mask, i], bins=50, alpha=0.5, 
                    label=f'Other categories', density=True)
    
    plt.xlabel(f'Probability for {category_names[i]}')
    plt.ylabel('Density')
    plt.title(f'Probability Distribution: {category_names[i]}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / "probability_histograms.png", dpi=300, bbox_inches='tight')


# Confusion Matrix
plt.figure(figsize=(20, 10))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, sample_weight=w_test)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix (counts)
plt.subplot(1, 2, 1)
im1 = plt.imshow(cm, cmap='Blues', aspect='auto')
plt.colorbar(im1)
plt.title('Confusion Matrix (Counts)')
plt.xlabel('Predicted')
plt.ylabel('True')

# Add text annotations
for i in range(n_categories):
    for j in range(n_categories):
        plt.text(j, i, str(int(cm[i, j])), 
                ha='center', va='center', fontsize=8)

plt.xticks(range(n_categories), [category_names[i] for i in range(n_categories)], rotation=45)
plt.yticks(range(n_categories), [category_names[i] for i in range(n_categories)])

# Plot confusion matrix (normalized)
plt.subplot(1, 2, 2)
im2 = plt.imshow(cm_normalized, cmap='Blues', aspect='auto')
plt.colorbar(im2)
plt.title('Confusion Matrix (Normalized)')
plt.xlabel('Predicted')
plt.ylabel('True')

# Add text annotations
for i in range(n_categories):
    for j in range(n_categories):
        plt.text(j, i, f'{cm_normalized[i, j]:.2f}', 
                ha='center', va='center', fontsize=8)

plt.xticks(range(n_categories), [category_names[i] for i in range(n_categories)], rotation=45)
plt.yticks(range(n_categories), [category_names[i] for i in range(n_categories)])

plt.tight_layout()
plt.savefig(plots_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')


#%%