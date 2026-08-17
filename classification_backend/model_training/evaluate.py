import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


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
    "classifier_validation_dataset.csv",
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

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Validation Dataset")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)

print(f"Dataset Shape : {df.shape}")

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())


# ==========================================================
# IMPORTANT
# Map the validation labels to the classifier labels.
#
# Update this mapping according to the meaning of
# Class = 1,2,3,4 in the validation dataset.
# ==========================================================

RISK_MAPPING = {

    1: "High",

    2: "Intermediate",

    3: "Low",

    4: "Low",

}

df["TrueRisk"] = df["RiskClass"].map(RISK_MAPPING)


print("\nMapped Risk Distribution")

print(df["TrueRisk"].value_counts())


# ==========================================================
# Features
# ==========================================================

FEATURE_COLUMNS = joblib.load(
    os.path.join(
        MODEL_DIR,
        "feature_columns.pkl",
    )
)

encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl",
    )
)

classifier = joblib.load(
    os.path.join(
        MODEL_DIR,
        "classifier.pkl",
    )
)


X = df[
    FEATURE_COLUMNS
]

y_true = df["TrueRisk"]


print("\nFeatures")

for feature in FEATURE_COLUMNS:

    print(feature)


# ==========================================================
# Prediction
# ==========================================================

print("\nRunning Classifier...\n")

encoded_predictions = classifier.predict(X)

prediction_probability = classifier.predict_proba(X)

predicted_labels = encoder.inverse_transform(
    encoded_predictions
)

df["PredictedRisk"] = predicted_labels


# ==========================================================
# Probability Columns
# ==========================================================

for i, cls in enumerate(encoder.classes_):

    df[f"Probability_{cls}"] = prediction_probability[:, i]


# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(
    y_true,
    predicted_labels,
)

precision = precision_score(
    y_true,
    predicted_labels,
    average="macro",
    zero_division=0,
)

recall = recall_score(
    y_true,
    predicted_labels,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    y_true,
    predicted_labels,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_true,
    predicted_labels,
    average="weighted",
    zero_division=0,
)

report = classification_report(
    y_true,
    predicted_labels,
    zero_division=0,
)

matrix = confusion_matrix(
    y_true,
    predicted_labels,
    labels=encoder.classes_,
)

# ==========================================================
# Print Metrics
# ==========================================================

print("\n" + "=" * 60)
print("Validation Results")
print("=" * 60)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")

print("\nClassification Report")

print(report)

print("\nConfusion Matrix")

matrix_df = pd.DataFrame(
    matrix,
    index=encoder.classes_,
    columns=encoder.classes_,
)

print(matrix_df)


# ==========================================================
# Prediction Distribution
# ==========================================================

print("\nPrediction Distribution")

print(
    df["PredictedRisk"].value_counts()
)

print("\nTrue Distribution")

print(
    df["TrueRisk"].value_counts()
)


# ==========================================================
# Correct / Incorrect Predictions
# ==========================================================

df["Correct"] = (
    df["TrueRisk"] ==
    df["PredictedRisk"]
)

print("\nCorrect Predictions")

print(df["Correct"].value_counts())


# ==========================================================
# Save Predictions
# ==========================================================

prediction_csv = os.path.join(
    OUTPUT_DIR,
    "classifier_validation_predictions.csv",
)

df.to_csv(
    prediction_csv,
    index=False,
)


# ==========================================================
# Save Confusion Matrix
# ==========================================================

confusion_csv = os.path.join(
    OUTPUT_DIR,
    "classifier_validation_confusion_matrix.csv",
)

matrix_df.to_csv(
    confusion_csv,
)


# ==========================================================
# Save Metrics
# ==========================================================

metrics = pd.DataFrame({

    "Metric": [

        "Accuracy",
        "Precision",
        "Recall",
        "Macro_F1",
        "Weighted_F1",

    ],

    "Value": [

        accuracy,
        precision,
        recall,
        macro_f1,
        weighted_f1,

    ]

})

metrics_csv = os.path.join(
    OUTPUT_DIR,
    "classifier_validation_metrics.csv",
)

metrics.to_csv(
    metrics_csv,
    index=False,
)


# ==========================================================
# Save Report
# ==========================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "classifier_validation_report.txt",
)

with open(report_file, "w") as f:

    f.write("=" * 80 + "\n")
    f.write("Classifier External Validation\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Accuracy    : {accuracy:.4f}\n")
    f.write(f"Precision   : {precision:.4f}\n")
    f.write(f"Recall      : {recall:.4f}\n")
    f.write(f"Macro F1    : {macro_f1:.4f}\n")
    f.write(f"Weighted F1 : {weighted_f1:.4f}\n\n")

    f.write("Classification Report\n")
    f.write(report)
    f.write("\n\n")

    f.write("Confusion Matrix\n")
    f.write(matrix_df.to_string())
    f.write("\n")


# ==========================================================
# Show Sample Predictions
# ==========================================================

print("\nSample Predictions")

preview_columns = [
    "Medication",
    "dose_nm",
    "TrueRisk",
    "PredictedRisk",
]

for cls in encoder.classes_:
    preview_columns.append(f"Probability_{cls}")

print(
    df[preview_columns].head(20)
)


# ==========================================================
# Finished
# ==========================================================

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)

print("\nSaved Files")

print(prediction_csv)
print(confusion_csv)
print(metrics_csv)
print(report_file)

print("\nDone.")