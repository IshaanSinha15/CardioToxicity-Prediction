import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedKFold,
)

from sklearn.preprocessing import LabelEncoder

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

df = pd.read_csv(INPUT_CSV)

print(df.shape)


# ==========================================================
# Features
# ==========================================================

FEATURE_COLUMNS = [
    "dose_nm",
    "Block_IKr",
    "Block_INa",
    "Block_ICaL",
]

TARGET_COLUMN = "RiskClass"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]
groups = df["smiles"]


# ==========================================================
# Encode Labels
# ==========================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nClasses")

for i, cls in enumerate(encoder.classes_):
    print(i, cls)

print("\nDistribution")

print(y.value_counts())


# ==========================================================
# Molecule Split
# ==========================================================

splitter = GroupShuffleSplit(
    test_size=0.20,
    n_splits=1,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y_encoded,
        groups=groups,
    )
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y_encoded[train_idx]
y_test = y_encoded[test_idx]

train_groups = groups.iloc[train_idx]
test_groups = groups.iloc[test_idx]

assert len(
    set(train_groups) &
    set(test_groups)
) == 0

print("\nTrain :", len(X_train))
print("Test :", len(X_test))


# ==========================================================
# Balanced Training Set
# ==========================================================

train_df = X_train.copy()
train_df["label"] = y_train

counts = train_df["label"].value_counts()

print("\nOriginal Training Distribution")

print(counts)

# Use SECOND largest class size instead of largest
target_size = counts.sort_values(
    ascending=False
).iloc[1]

balanced = []

for cls in sorted(train_df["label"].unique()):

    subset = train_df[
        train_df["label"] == cls
    ]

    if len(subset) < target_size:

        subset = resample(
            subset,
            replace=True,
            n_samples=target_size,
            random_state=42,
        )

    elif len(subset) > target_size:

        subset = resample(
            subset,
            replace=False,
            n_samples=target_size,
            random_state=42,
        )

    balanced.append(subset)

balanced_df = pd.concat(
    balanced,
    ignore_index=True,
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42,
)

X_train = balanced_df[
    FEATURE_COLUMNS
]

y_train = balanced_df[
    "label"
]

print("\nBalanced Distribution")

print(
    pd.Series(y_train).value_counts()
)


# ==========================================================
# Save Metadata
# ==========================================================

joblib.dump(
    FEATURE_COLUMNS,
    os.path.join(
        MODEL_DIR,
        "feature_columns.pkl",
    ),
)

joblib.dump(
    encoder,
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl",
    ),
)


# ==========================================================
# Evaluation
# ==========================================================

def evaluate(model):

    pred = model.predict(X_test)

    return {

        "Accuracy":
            accuracy_score(
                y_test,
                pred,
            ),

        "Precision":
            precision_score(
                y_test,
                pred,
                average="macro",
                zero_division=0,
            ),

        "Recall":
            recall_score(
                y_test,
                pred,
                average="macro",
                zero_division=0,
            ),

        "Macro_F1":
            f1_score(
                y_test,
                pred,
                average="macro",
                zero_division=0,
            ),

        "Weighted_F1":
            f1_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0,
            ),

        "Report":
            classification_report(
                y_test,
                pred,
                target_names=encoder.classes_,
                zero_division=0,
            ),

        "Matrix":
            confusion_matrix(
                y_test,
                pred,
            ),
    }


# ==========================================================
# Models
# ==========================================================

models = {

    "Random Forest":

        RandomForestClassifier(

            random_state=42,

            n_jobs=-1,

            class_weight="balanced",

        ),

    "XGBoost":

        XGBClassifier(

            random_state=42,

            eval_metric="mlogloss",

            tree_method="hist",

            n_jobs=-1,

        ),

}


# ==========================================================
# Parameter Search
# ==========================================================

param_grids = {

    "Random Forest": {

        "n_estimators": [200, 300],

        "max_depth": [15, 25],

        "min_samples_split": [2, 5],

    },

    "XGBoost": {

        "n_estimators": [300, 500],

        "max_depth": [5, 6],

        "learning_rate": [0.05, 0.1],

        "subsample": [0.8, 1.0],

    },

}


cv = StratifiedKFold(

    n_splits=3,

    shuffle=True,

    random_state=42,

)

results = {}

best_model = None

best_name = None

best_score = -1

# ==========================================================
# Training
# ==========================================================

print("\n" + "=" * 60)
print("Training Models")
print("=" * 60)

for name, model in models.items():

    print(f"\n{name}")
    print("-" * 50)

    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=param_grids[name],

        n_iter=5,

        scoring="f1_macro",

        cv=cv,

        random_state=42,

        n_jobs=-1,

        verbose=1,

    )

    search.fit(
        X_train,
        y_train,
    )

    best = search.best_estimator_

    print("\nBest Parameters")
    print(search.best_params_)

    print(f"\nBest CV Macro F1 : {search.best_score_:.4f}")

    evaluation = evaluate(best)

    results[name] = {

        "Model": best,

        "CV": search.best_score_,

        **evaluation,

    }

    print("\nAccuracy")
    print(f"{evaluation['Accuracy']:.4f}")

    print("\nMacro F1")
    print(f"{evaluation['Macro_F1']:.4f}")

    print("\nWeighted F1")
    print(f"{evaluation['Weighted_F1']:.4f}")

    print("\nClassification Report")
    print(evaluation["Report"])

    print("\nConfusion Matrix")
    print(evaluation["Matrix"])

    if evaluation["Macro_F1"] > best_score:

        best_score = evaluation["Macro_F1"]
        best_model = best
        best_name = name


# ==========================================================
# Save Best Model
# ==========================================================

joblib.dump(

    best_model,

    os.path.join(
        MODEL_DIR,
        "classifier.pkl",
    ),

)

print("\nBest Model Saved")
print(best_name)


# ==========================================================
# Feature Importance
# ==========================================================

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": FEATURE_COLUMNS,

        "Importance": best_model.feature_importances_,

    })

    importance = importance.sort_values(

        "Importance",

        ascending=False,

    )

    importance.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv",
        ),

        index=False,

    )

    print("\nFeature Importance")

    print(importance)

else:

    print("\nSelected model does not expose feature importances.")


# ==========================================================
# Model Comparison
# ==========================================================

comparison = []

for name, result in results.items():

    comparison.append({

        "Model": name,

        "Accuracy": result["Accuracy"],

        "Precision": result["Precision"],

        "Recall": result["Recall"],

        "Macro_F1": result["Macro_F1"],

        "Weighted_F1": result["Weighted_F1"],

        "CV_Macro_F1": result["CV"],

    })

comparison = pd.DataFrame(comparison)

comparison = comparison.sort_values(

    "Macro_F1",

    ascending=False,

)

comparison.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv",
    ),

    index=False,

)

print("\nModel Comparison")
print(comparison)


# ==========================================================
# Save Reports
# ==========================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_reports.txt",
)

with open(report_path, "w") as f:

    for name, result in results.items():

        f.write("=" * 80 + "\n")
        f.write(f"{name}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"CV Macro F1 : {result['CV']:.4f}\n")
        f.write(f"Accuracy    : {result['Accuracy']:.4f}\n")
        f.write(f"Precision   : {result['Precision']:.4f}\n")
        f.write(f"Recall      : {result['Recall']:.4f}\n")
        f.write(f"Macro F1    : {result['Macro_F1']:.4f}\n")
        f.write(f"Weighted F1 : {result['Weighted_F1']:.4f}\n\n")

        f.write("Classification Report\n")
        f.write(result["Report"])
        f.write("\n\n")

        f.write("Confusion Matrix\n")
        f.write(str(result["Matrix"]))
        f.write("\n\n")


# ==========================================================
# Final Summary
# ==========================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"\nBest Model : {best_name}")
print(f"Macro F1   : {best_score:.4f}")

print("\nSaved Models")

print(os.path.join(
    MODEL_DIR,
    "classifier.pkl",
))

print(os.path.join(
    MODEL_DIR,
    "label_encoder.pkl",
))

print(os.path.join(
    MODEL_DIR,
    "feature_columns.pkl",
))

print("\nSaved Outputs")

print(os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv",
))

print(os.path.join(
    OUTPUT_DIR,
    "classification_reports.txt",
))

if hasattr(best_model, "feature_importances_"):

    print(os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv",
    ))

print("\nDone.")