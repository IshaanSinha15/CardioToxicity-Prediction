import os
import pandas as pd

from prediction_backend.inference.predict import predict
from classification_backend.dose_response.hill_equation import HillEquation


# ==========================================================
# Dose Panel (nM)
# ==========================================================

DOSE_PANEL = [
    0.3,
    1,
    3,
    10,
    30,
    100,
    300,
    1000,
    3000,
]


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
    "combined_ic50_dataset.csv",
)

OUTPUT_CSV = os.path.join(
    ROOT_DIR,
    "classification_backend",
    "dataset",
    "classifier_dataset.csv",
)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Complete IC50 Dataset")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)

print(f"Dataset Shape : {df.shape}")

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())


# ==========================================================
# Prediction Cache
# ==========================================================

prediction_cache = {}


# ==========================================================
# Statistics
# ==========================================================

predicted_ikr = 0
predicted_ina = 0
predicted_ical = 0

rows = []


# ==========================================================
# Helper Function
# ==========================================================

def get_complete_ic50(drug):
    """
    Returns complete IC50 values (nM) for all channels.
    Missing values are predicted using the regression pipeline.
    """

    global predicted_ikr
    global predicted_ina
    global predicted_ical

    smiles = drug["smiles"]

    ikr = pd.to_numeric(drug["IC50_IKr"], errors="coerce")
    ina = pd.to_numeric(drug["IC50_INa"], errors="coerce")
    ical = pd.to_numeric(drug["IC50_ICaL"], errors="coerce")

    ikr_source = "Dataset"
    ina_source = "Dataset"
    ical_source = "Dataset"

    # ------------------------------------------------------
    # Predict only if any channel is missing
    # ------------------------------------------------------

    if pd.isna(ikr) or pd.isna(ina) or pd.isna(ical):

        if smiles not in prediction_cache:

            try:
                prediction_cache[smiles] = predict(smiles)

            except Exception as e:

                print(f"\nPrediction failed for molecule:")
                print(smiles)
                print(e)

                return None

        preds = prediction_cache[smiles]

        if pd.isna(ikr):
            ikr = preds["herg"]["IC50_nM"]
            ikr_source = "Predicted"
            predicted_ikr += 1

        if pd.isna(ina):
            ina = preds["nav"]["IC50_nM"]
            ina_source = "Predicted"
            predicted_ina += 1

        if pd.isna(ical):
            ical = preds["cav"]["IC50_nM"]
            ical_source = "Predicted"
            predicted_ical += 1

    return {
        "smiles": smiles,

        "IC50_IKr": float(ikr),
        "IC50_INa": float(ina),
        "IC50_ICaL": float(ical),

        "IKr_Source": ikr_source,
        "INa_Source": ina_source,
        "ICaL_Source": ical_source,

        "ic50_source": drug["ic50_source"],
    }


# ==========================================================
# Start Dataset Generation
# ==========================================================

print("\n" + "=" * 60)
print("Generating Classifier Dataset")
print("=" * 60)

# ==========================================================
# Process Every Molecule
# ==========================================================

for index, drug in df.iterrows():

    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1}/{len(df)} molecules...")

    complete = get_complete_ic50(drug)

    if complete is None:
        continue

    smiles = complete["smiles"]

    ikr_ic50 = complete["IC50_IKr"]
    ina_ic50 = complete["IC50_INa"]
    ical_ic50 = complete["IC50_ICaL"]

    # ------------------------------------------------------
    # Create Hill Equation Models
    # ------------------------------------------------------

    ikr_model = HillEquation(ic50_nm=ikr_ic50)
    ina_model = HillEquation(ic50_nm=ina_ic50)
    ical_model = HillEquation(ic50_nm=ical_ic50)

    # ------------------------------------------------------
    # Generate one sample for every dose
    # ------------------------------------------------------

    for dose in DOSE_PANEL:

        block_ikr = ikr_model.block(dose)
        block_ina = ina_model.block(dose)
        block_ical = ical_model.block(dose)

        rows.append({

            # -----------------------------
            # Molecule
            # -----------------------------

            "smiles": smiles,

            # -----------------------------
            # Dose
            # -----------------------------

            "dose_nm": dose,

            # -----------------------------
            # Complete IC50 values
            # -----------------------------

            "IC50_IKr": ikr_ic50,
            "IC50_INa": ina_ic50,
            "IC50_ICaL": ical_ic50,

            # -----------------------------
            # Source Tracking
            # -----------------------------

            "IKr_Source": complete["IKr_Source"],
            "INa_Source": complete["INa_Source"],
            "ICaL_Source": complete["ICaL_Source"],

            "ic50_source": complete["ic50_source"],

            # -----------------------------
            # Dose Response
            # -----------------------------

            "Block_IKr": block_ikr,
            "Block_INa": block_ina,
            "Block_ICaL": block_ical,

        })


# ==========================================================
# Save Dataset
# ==========================================================

classifier_df = pd.DataFrame(rows)

classifier_df.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 60)
print("Classifier Dataset Generated")
print("=" * 60)

print(f"Total Molecules : {df['smiles'].nunique()}")
print(f"Total Samples   : {len(classifier_df)}")
print(f"Dose Levels     : {len(DOSE_PANEL)}")

print("\nPrediction Summary")

print(f"IKr Predicted  : {predicted_ikr}")
print(f"INa Predicted  : {predicted_ina}")
print(f"ICaL Predicted : {predicted_ical}")

print("\nPrediction Cache Size")
print(len(prediction_cache))

print("\nColumns")
print(classifier_df.columns.tolist())

print("\nMissing Values")
print(classifier_df.isnull().sum())

print("\nDose Distribution")
print(classifier_df["dose_nm"].value_counts().sort_index())

print("\nIC50 Summary")
print(
    classifier_df[
        [
            "IC50_IKr",
            "IC50_INa",
            "IC50_ICaL",
        ]
    ].describe()
)

print("\nChannel Block Summary")
print(
    classifier_df[
        [
            "Block_IKr",
            "Block_INa",
            "Block_ICaL",
        ]
    ].describe()
)

print("\nIC50 Source Summary")

print("\nIKr Source")
print(classifier_df["IKr_Source"].value_counts())

print("\nINa Source")
print(classifier_df["INa_Source"].value_counts())

print("\nICaL Source")
print(classifier_df["ICaL_Source"].value_counts())

print("\nSaved To")
print(OUTPUT_CSV)

print("\nDataset generation completed successfully.")