import pandas as pd
import re

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GroupKFold,
    cross_val_score,
    GridSearchCV,
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.utils import resample


# ==========================================================
# Paths
# ==========================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

INPUT_CSV = os.path.join(
    ROOT_DIR,
    "classification_backend",
    "dataset",
    "classifier_dataset_labeled.csv",
)

MODEL_DIR = os.path.join(
    ROOT_DIR,
    "classification_backend",
    "saved_models",
)

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "classification_backend",
    "outputs",
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Classifier Dataset")
print("=" * 60)

# ==========================================================
# Load Data
# ==========================================================
# train_pool_20k_final.csv: 20,000 balanced rows built with independent
#                         per-channel interpolation (v2 generator) -- less
#                         artificial collinearity than v1.
# holdout_unseen_drugs_final.csv: real drugs held out before synthetic
#                         generation, never used as a blend parent. Touched
#                         once, at the very end.

df = pd.read_csv("data/datasets/train_pool_20k_final.csv")
df_holdout = pd.read_csv("data/datasets/holdout_unseen_drugs_final.csv")

print(df.shape)
print(df.head().to_string(index=False))
print(df["Class"].value_counts().sort_index())

print("\nHoldout (unseen drugs) shape:", df_holdout.shape)
print(df_holdout["Class"].value_counts().sort_index())

# ==========================================================
# Encode Labels
# ==========================================================
# APA is dropped: it's an exact identity (Peak - RMP), not an independent
# measurement. Keeping it adds zero real signal and only gives the model
# extra room to fit noise/artifacts.
# Bnet is dropped: it's the exact formula the Class label was derived from
# (0.5*(log10(IC50_ICaL/EFTPC)+log10(IC50_INa/EFTPC)) - log10(IC50_IKr/EFTPC),
# then quartile-binned). Keeping it in X would let the model just learn the
# label-generation formula instead of learning from the raw channel data
# (IC50_IKr, IC50_INa, IC50_ICaL, Block_*) the way you actually want it to.

DROP_COLS = ["Medication", "Class", "Bnet"]

X = df.drop(columns=DROP_COLS)
y = df["Class"]

X_holdout = df_holdout.drop(columns=DROP_COLS)
y_holdout = df_holdout["Class"]

print("\nFeature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)
print("\nFeature Columns:")
print(X.columns.tolist())
print("\nTarget Classes:")
print(sorted(y.unique()))

# ==========================================================
# Group labels for leakage-aware CV
# ==========================================================
# Every synthetic row is named "DrugA_x_DrugB_synthN". Two rows built from
# the same two parent drugs are near-duplicates -- if a plain KFold splits
# them across train/val folds, the CV score becomes optimistic in the same
# way the old row-level train_test_split was. GroupKFold below groups rows
# by their *sorted parent-drug pair* so an entire family stays on one side
# of every fold, giving a CV estimate that's actually predictive of the
# unseen-drug holdout score, instead of just remeasuring memorization.
#
# Real (non-synthetic) rows get their own drug name as their own group,
# since they have no "parents" to leak across.


def parent_group(name: str) -> str:
    if "_synth" in name:
        base = name.split("_synth")[0]
        parents = sorted(base.split("_x_"))
        return "|".join(parents)
    return name


groups = df["Medication"].apply(parent_group)
print("\nNumber of distinct drug-family groups in train_pool:", groups.nunique())

# ==========================================================
# Train-Test Split (grouped, within train_pool_20k_final.csv)
# ==========================================================
# Using GroupShuffleSplit-equivalent behavior via a manual grouped split so
# the internal test set doesn't share a parent-drug family with the internal
# training set either. This makes "internal test split accuracy" a much more
# honest number than before, closer to (though still not a substitute for)
# the unseen holdout.

from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

print("\nGrouped Train-Test Split (within train_pool_20k_final.csv)")
print("-" * 40)
print("Training samples :", X_train.shape[0])
print("Testing samples  :", X_test.shape[0])
print("Distinct families in train:", groups_train.nunique())
print("Distinct families in test :", groups.iloc[test_idx].nunique())

# ==========================================================
# Balanced Training Set
# ==========================================================
# Two changes from before:
#   1. GroupKFold (grouped by parent-drug family) instead of plain
#      StratifiedKFold/cv=5 int, so hyperparameter selection itself can't
#      exploit synthetic-blend leakage the way it did previously (that's
#      how max_depth=None got picked last time -- the CV loop couldn't see
#      the overfitting because both sides of every fold could share
#      parents).
#   2. A regularized grid: max_depth caps below None, larger
#      min_samples_leaf, and max_features to decorrelate trees given the
#      collinear feature set.

param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [4, 6, 8, 10],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [4, 8, 16],
    "max_features": ["sqrt", 0.5],
    "class_weight": ["balanced"],
}

group_cv = GroupKFold(n_splits=5)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=group_cv,
    scoring="accuracy",
    n_jobs=-1,
)

grid_search.fit(X_train, y_train, groups=groups_train)

rf_model = grid_search.best_estimator_

    print("\nBest Parameters")
    print(search.best_params_)

print("Best CV Score (grouped, train_pool_20k_final.csv only):")
print(grid_search.best_score_)

print("\nRandom Forest Model:")
print(rf_model)

# ==========================================================
# Feature Importance
# ==========================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print("\nFeature Importance")
print(feature_importance.to_string(index=False))

print("\nFeature Correlation Matrix")
print(X.corr().round(2).to_string())

# ==========================================================
# Prediction on the internal (grouped) test split
# ==========================================================

y_pred = rf_model.predict(X_test)
print("Predictions generated successfully (internal grouped test split)")

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation -- Internal Grouped Test Split (train_pool_20k_final.csv)")
print("-" * 40)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report (internal grouped test split):")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix (internal grouped test split):")
print(confusion_matrix(y_test, y_pred))

# ==========================================================
# FINAL Model Evaluation -- Unseen Drug Holdout
# ==========================================================
# Runs once, after the final model is already selected above. Do not loop
# back and re-tune based on these numbers.

print("\n")
print("=" * 60)
print("FINAL Evaluation on Unseen Drug Holdout")
print("=" * 60)

y_holdout_pred = rf_model.predict(X_holdout)
holdout_accuracy = accuracy_score(y_holdout, y_holdout_pred)

print(f"Unseen-drug holdout accuracy: {holdout_accuracy:.4f}")
print(f"(compare against internal grouped test split accuracy: {accuracy:.4f})")

print("\nClassification Report (unseen drug holdout):")
print(classification_report(y_holdout, y_holdout_pred, zero_division=0))
print("\nConfusion Matrix (unseen drug holdout):")
print(confusion_matrix(y_holdout, y_holdout_pred))

print("\nPer-drug predictions (unseen holdout):")
holdout_report = df_holdout[["Medication", "Class"]].copy()
holdout_report["Predicted"] = y_holdout_pred
holdout_report["Correct"] = holdout_report["Class"] == holdout_report["Predicted"]
print(holdout_report.to_string(index=False))

gap = accuracy - holdout_accuracy
print(f"\nGeneralization gap (internal grouped split - unseen holdout): {gap:.4f}")
print("If this gap is still large after grouping + regularizing, the model")
print("has hit the information ceiling of the ~86 real seen drugs -- the")
print("fix at that point is more real labeled drugs (e.g. running the 28")
print("CiPA drugs through your real ORDSimulator), not more synthetic rows.")

# ==========================================================
# Model Comparison
# ==========================================================

model_bundle = {
    "model": rf_model,
    "feature_names": list(X.columns),
    "classes": list(rf_model.classes_),
    "n_features": X.shape[1],
}

joblib.dump(model_bundle, "saved_models/random_forest_classifier.pkl")

print("\nBest model saved successfully!")
print("Model bundle: saved_models/random_forest_classifier.pkl")

# ==========================================================
# Cross Validation (grouped, train_pool_20k_final.csv only)
# ==========================================================
# Reported for comparison against the old ungrouped StratifiedKFold score --
# expect this to be noticeably lower than 0.977 if grouping is doing its job.

print("\n")
print("=" * 60)
print("Random Forest 5-Fold GROUPED Cross Validation (train_pool_20k_final.csv)")
print("=" * 60)

cv_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=GroupKFold(n_splits=5),
    groups=groups,
    scoring="accuracy",
)

print("Fold Accuracies:", cv_scores)
print(f"Mean Accuracy : {cv_scores.mean():.4f}")
print(f"Std Deviation : {cv_scores.std():.4f}")

print("\n")
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Internal grouped test split accuracy : {accuracy:.4f}")
print(f"5-fold GROUPED CV mean accuracy       : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"Unseen-drug holdout accuracy          : {holdout_accuracy:.4f}  <-- most trustworthy number")
