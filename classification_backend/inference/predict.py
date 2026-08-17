import os
import joblib
import numpy as np
import pandas as pd

from prediction_backend.inference.predict import predict


# ==========================================================
# Paths
# ==========================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

DATASET = os.path.join(
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


# ==========================================================
# Load Models
# ==========================================================

classifier = joblib.load(
    os.path.join(
        MODEL_DIR,
        "classifier.pkl",
    )
)

encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl",
    )
)

feature_columns = joblib.load(
    os.path.join(
        MODEL_DIR,
        "feature_columns.pkl",
    )
)


dataset = pd.read_csv(DATASET)
dataset["smiles"] = dataset["smiles"].astype(str)


# ==========================================================
# Hill Equation
# ==========================================================

def block_percent(
    dose_nm,
    ic50_nm,
):
    return (
        dose_nm /
        (dose_nm + ic50_nm)
    ) * 100


# ==========================================================
# Dominant Channel
# ==========================================================

def dominant_channel(
    ikr,
    ina,
    ical,
):

    values = {

        "IKr": ikr,
        "INa": ina,
        "ICaL": ical,

    }

    return max(
        values,
        key=values.get,
    )


# ==========================================================
# Explain Prediction
# ==========================================================

def explain_prediction(
    ikr,
    ina,
    ical,
):

    explanation = []

    # IKr

    if ikr >= 50:
        explanation.append("Strong IKr block")

    elif ikr >= 20:
        explanation.append("Moderate IKr block")

    elif ikr >= 5:
        explanation.append("Mild IKr block")


    # INa

    if ina >= 50:
        explanation.append("Strong INa block")

    elif ina >= 20:
        explanation.append("Moderate INa block")

    elif ina >= 5:
        explanation.append("Mild INa block")


    # ICaL

    if ical >= 50:
        explanation.append("Strong ICaL block")

    elif ical >= 20:
        explanation.append("Moderate ICaL block")

    elif ical >= 5:
        explanation.append("Mild ICaL block")


    if len(explanation) == 0:

        explanation.append(
            "Minimal ion channel inhibition"
        )

    return explanation


# ==========================================================
# Dataset Lookup
# ==========================================================

def lookup_dataset(
    smiles,
    dose,
):

    compound = dataset[
        dataset["smiles"] == smiles
    ]

    if len(compound) == 0:

        return None, None

    doses = sorted(
        compound["dose_nm"].unique()
    )

    idx = (
        abs(
            compound["dose_nm"] - dose
        )
    ).idxmin()

    row = compound.loc[idx]

    return row, doses


# ==========================================================
# Interactive CLI
# ==========================================================

print("=" * 60)
print("Cardiotoxicity Classification Predictor")
print("=" * 60)

while True:

    smiles = input(
        "\nEnter SMILES (or exit): "
    ).strip()

    if smiles.lower() == "exit":
        break

    try:

        dose = float(
            input(
                "Enter Dose (nM): "
            )
        )

    except ValueError:

        print("Invalid dose.")

        continue


    # ------------------------------------------------------
    # Regression Prediction
    # ------------------------------------------------------

    try:

        regression = predict(smiles)

    except Exception as e:

        print("\nPrediction Failed")

        print(e)

        continue


    ikr_ic50 = regression["herg"]["IC50_nM"]
    ina_ic50 = regression["nav"]["IC50_nM"]
    ical_ic50 = regression["cav"]["IC50_nM"]


    # ------------------------------------------------------
    # Channel Block
    # ------------------------------------------------------

    block_ikr = block_percent(
        dose,
        ikr_ic50,
    )

    block_ina = block_percent(
        dose,
        ina_ic50,
    )

    block_ical = block_percent(
        dose,
        ical_ic50,
    )


    # ------------------------------------------------------
    # Classifier Prediction
    # ------------------------------------------------------

    features = pd.DataFrame(

        [[

            dose,
            block_ikr,
            block_ina,
            block_ical,

        ]],

        columns=feature_columns,

    )

    encoded = classifier.predict(
        features
    )[0]

    probabilities = classifier.predict_proba(
        features
    )[0]

    predicted_label = encoder.inverse_transform(
        [encoded]
    )[0]


    # ------------------------------------------------------
    # Dataset Lookup
    # ------------------------------------------------------

    actual, doses = lookup_dataset(
        smiles,
        dose,
    )
    # ======================================================
    # Regression Prediction
    # ======================================================

    print("\n" + "=" * 60)
    print("Regression Prediction")
    print("=" * 60)

    print(f"IKr IC50  : {ikr_ic50:.2f} nM")
    print(f"INa IC50  : {ina_ic50:.2f} nM")
    print(f"ICaL IC50 : {ical_ic50:.2f} nM")


    # ======================================================
    # Channel Block
    # ======================================================

    print("\n" + "=" * 60)
    print("Channel Block")
    print("=" * 60)

    print(f"Dose : {dose:.2f} nM\n")

    print(f"IKr  : {block_ikr:.2f}%")
    print(f"INa  : {block_ina:.2f}%")
    print(f"ICaL : {block_ical:.2f}%")

    print(
        f"\nDominant Channel : "
        f"{dominant_channel(block_ikr, block_ina, block_ical)}"
    )


    # ======================================================
    # Prediction Explanation
    # ======================================================

    print("\n" + "=" * 60)
    print("Prediction Explanation")
    print("=" * 60)

    for item in explain_prediction(
        block_ikr,
        block_ina,
        block_ical,
    ):
        print(f"• {item}")


    # ======================================================
    # Classifier Output
    # ======================================================

    print("\n" + "=" * 60)
    print("Classifier Prediction")
    print("=" * 60)

    print(f"Predicted Risk : {predicted_label}")

    print("\nClass Probabilities")

    for cls, prob in zip(
        encoder.classes_,
        probabilities,
    ):
        print(f"{cls:<15}: {prob * 100:.2f}%")


    # ======================================================
    # Dataset Comparison
    # ======================================================

    print("\n" + "=" * 60)
    print("Dataset Lookup")
    print("=" * 60)

    if actual is None:

        print("Compound not found in dataset.")

    else:

        print("Compound Found")

        print("\nAvailable Doses")

        for d in doses:
            print(f"{d} nM")

        print(f"\nRequested Dose : {dose:.2f} nM")
        print(f"Closest Dose   : {actual['dose_nm']:.2f} nM")

        print(f"\nActual Risk : {actual['RiskClass']}")

        print("\nDataset IC50")

        print(f"IKr  : {actual['IC50_IKr']:.2f} nM")
        print(f"INa  : {actual['IC50_INa']:.2f} nM")
        print(f"ICaL : {actual['IC50_ICaL']:.2f} nM")

        print("\nDataset Block")

        print(f"IKr  : {actual['Block_IKr']:.2f}%")
        print(f"INa  : {actual['Block_INa']:.2f}%")
        print(f"ICaL : {actual['Block_ICaL']:.2f}%")

        print("\nIC50 Sources")

        print(f"IKr  : {actual['IKr_Source']}")
        print(f"INa  : {actual['INa_Source']}")
        print(f"ICaL : {actual['ICaL_Source']}")

        print("\nRegression Error")

        print(
            f"IKr  : {abs(ikr_ic50 - actual['IC50_IKr']):.2f} nM"
        )

        print(
            f"INa  : {abs(ina_ic50 - actual['IC50_INa']):.2f} nM"
        )

        print(
            f"ICaL : {abs(ical_ic50 - actual['IC50_ICaL']):.2f} nM"
        )

        print("\nPrediction Result")

        if predicted_label == actual["RiskClass"]:

            print("✓ Correct Prediction")

        else:

            print("✗ Incorrect Prediction")


    # ======================================================
    # Summary
    # ======================================================

    print("\n" + "=" * 60)
    print("Prediction Summary")
    print("=" * 60)

    print(f"SMILES         : {smiles}")
    print(f"Dose           : {dose:.2f} nM")

    print(f"\nPredicted Risk : {predicted_label}")

    if actual is not None:

        print(f"Actual Risk    : {actual['RiskClass']}")

    print(
        f"\nDominant Block : "
        f"{dominant_channel(block_ikr, block_ina, block_ical)}"
    )

    confidence = probabilities.max() * 100

    print(f"Confidence     : {confidence:.2f}%")

    print("\n" + "-" * 60)

print("\nExiting Predictor...")