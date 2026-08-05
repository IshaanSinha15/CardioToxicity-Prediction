import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import os
import joblib

# ==========================================================
# Load Data
# ==========================================================
# train_pool_20k.csv: 20,000 balanced rows (real "seen" drugs + synthetic
#                      blends of only those seen drugs). All tuning/CV happens
#                      here.
# holdout_unseen_drugs.csv: real drugs held out BEFORE synthetic generation,
#                      never used as a blend parent. Touched only once, at the
#                      very end, as a genuine unseen-drug evaluation.

df = pd.read_csv("data/datasets/train_pool_20k.csv")
df_holdout = pd.read_csv("data/datasets/holdout_unseen_drugs.csv")

print(df.shape)
print(df.head().to_string(index=False))
print(df["Class"].value_counts().sort_index())

print("\nHoldout (unseen drugs) shape:", df_holdout.shape)
print(df_holdout["Class"].value_counts().sort_index())

# ==========================================================
# Data Preprocessing
# ==========================================================

# Features (X) / Target (y) -- train pool
X = df.drop(columns=["Medication", "Class"])
y = df["Class"]

# Features (X) / Target (y) -- unseen holdout, kept completely separate
X_holdout = df_holdout.drop(columns=["Medication", "Class"])
y_holdout = df_holdout["Class"]

print("\nFeature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Classes:")
print(sorted(y.unique()))


# ==========================================================
# Train-Test Split (within train_pool_20k.csv only)
# ==========================================================
# NOTE: this split is for internal model selection / sanity-checking during
# development. It is NOT the unseen-drug evaluation -- both sides of this
# split are drawn from the same "seen" drug pool, so rows on either side can
# still be synthetic blends sharing a parent drug. The holdout set below is
# what tells you how the model does on drugs it has never seen in any form.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split (within train_pool_20k.csv)")
print("-" * 40)

print("Training samples :", X_train.shape[0])
print("Testing samples  :", X_test.shape[0])

print("\nTraining feature shape:", X_train.shape)
print("Testing feature shape :", X_test.shape)


# ==========================================================
# Random Forest Hyperparameter Tuning
# ==========================================================
# IMPORTANT: GridSearchCV's internal cross-validation (cv=5) is performed
# entirely on X_train/y_train, i.e. entirely within train_pool_20k.csv.
# The unseen holdout set is NEVER passed into fit(), never scored during
# tuning, and never used to pick hyperparameters. That's what keeps it a
# genuine unseen-drug test at the end.

param_grid = {
    "n_estimators": [200, 500, 800],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced"]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

rf_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("Best CV Score (train_pool_20k.csv only):")
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

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(feature_importance.to_string(index=False))

print("\nFeature Correlation Matrix")
print(X.corr().round(2).to_string())

# ==========================================================
# Prediction on the internal test split (train_pool_20k.csv)
# ==========================================================

y_pred = rf_model.predict(X_test)

print("Predictions generated successfully (internal test split)")

# ==========================================================
# Model Evaluation -- Internal Test Split
# ==========================================================
# This score is measured on rows drawn from the same "seen" drug pool as
# training, so treat it as an optimistic upper bound, not a generalization
# estimate. The real generalization estimate is the holdout section below.

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation -- Internal Test Split (train_pool_20k.csv)")
print("-" * 40)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report (internal test split):")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix (internal test split):")
print(confusion_matrix(y_test, y_pred))


# ==========================================================
# FINAL Model Evaluation -- Unseen Drug Holdout
# ==========================================================
# This is the evaluation that actually matters for judging generalization.
# df_holdout contains real drugs that were held out BEFORE any synthetic
# generation happened, so none of them contributed to any row the model was
# trained or tuned on, directly or as a blend parent. This block only runs
# ONCE, after the final model has already been selected above -- do not loop
# back and re-tune based on these numbers, or this stops being a valid
# unseen-drug estimate.

print("\n")
print("=" * 60)
print("FINAL Evaluation on Unseen Drug Holdout")
print("=" * 60)

y_holdout_pred = rf_model.predict(X_holdout)

holdout_accuracy = accuracy_score(y_holdout, y_holdout_pred)

print(f"Unseen-drug holdout accuracy: {holdout_accuracy:.4f}")
print(f"(compare against internal test split accuracy: {accuracy:.4f})")

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
print(f"\nGeneralization gap (internal split accuracy - unseen holdout accuracy): {gap:.4f}")
print("A large gap here means the model is overfitting to the synthetic")
print("interpolation structure rather than learning real class-separating")
print("pharmacology -- treat that as a signal to simplify the model")
print("(shallower trees / fewer estimators / stronger regularization),")
print("not as a reason to generate more synthetic rows.")


# ==========================================================
# Save Best Model
# ==========================================================

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
models_dir = os.path.join(repo_root, "saved_models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(rf_model, os.path.join(models_dir, "random_forest_classifier.pkl"))

print("\nBest model saved successfully!")
print("Model: saved_models/random_forest_classifier.pkl")


# ==========================================================
# Cross Validation (train_pool_20k.csv only)
# ==========================================================
# As with GridSearchCV above, this CV is run entirely on the seen-drug pool
# (X, y from train_pool_20k.csv). It is a stability check on the internal
# split, not a substitute for the unseen holdout evaluation above.

print("\n")
print("=" * 60)
print("Random Forest 5-Fold Cross Validation (train_pool_20k.csv)")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("Fold Accuracies:", scores)
print(f"Mean Accuracy : {scores.mean():.4f}")
print(f"Std Deviation : {scores.std():.4f}")

print("\n")
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Internal test split accuracy   : {accuracy:.4f}")
print(f"5-fold CV mean accuracy        : {scores.mean():.4f} (+/- {scores.std():.4f})")
print(f"Unseen-drug holdout accuracy   : {holdout_accuracy:.4f}  <-- most trustworthy number")