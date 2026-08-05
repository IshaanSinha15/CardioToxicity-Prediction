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

df = pd.read_csv("data/datasets/classification_train.csv")

print(df.shape)
print(df.head().to_string(index=False))
print(df["Class"].value_counts().sort_index())
# ==========================================================
# Data Preprocessing
# ==========================================================

# Features (X)
X = df.drop(columns=["Medication", "Class"])
# Target (y)
y = df["Class"]

print("\nFeature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Classes:")
print(sorted(y.unique()))


# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split")
print("-" * 40)

print("Training samples :", X_train.shape[0])
print("Testing samples  :", X_test.shape[0])

print("\nTraining feature shape:", X_train.shape)
print("Testing feature shape :", X_test.shape)


# ==========================================================
# Random Forest Hyperparameter Tuning
# ==========================================================

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

print("Best CV Score:")
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
# Prediction
# ==========================================================

y_pred = rf_model.predict(X_test)

print("Predictions generated successfully")

# ==========================================================
# Model Evaluation
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 40)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


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
# Cross Validation
# ==========================================================

print("\n")
print("=" * 60)
print("Random Forest 5-Fold Cross Validation")
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
