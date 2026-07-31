import os
import pandas as pd

from classification_backend.simulation.ord_simulator import ORDSimulator
from classification_backend.feature_extraction.ap_features import APFeatureExtractor


# ==========================================================
# Paths
# ==========================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

DATASET_DIR = os.path.join(ROOT_DIR, "data", "datasets")

INPUT_CSV = os.path.join(
    DATASET_DIR,
    "simulation_dataset.csv"
)

OUTPUT_CSV = os.path.join(
    DATASET_DIR,
    "classifier_dataset.csv"
)


# ==========================================================
# Read Dataset
# ==========================================================

df = pd.read_csv(INPUT_CSV)

numeric_columns = [
    "EFTPC",
    "IC50IKr", "hIKr",
    "IC50INa", "hiNa",
    "IC50INaL", "hINaL",
    "IC50ICaL", "hICaL",
    "IC50IKs", "hIKs",
    "IC50IK1", "hIK1",
    "IC50Ito", "hIto"
]

for col in numeric_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )

    df[col] = pd.to_numeric(df[col], errors="coerce")

print("=" * 60)
print("Simulation Dataset Loaded")
print("=" * 60)

print("Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================================
# Hill Equation
# ==========================================================

def hill_block(ic50, hill, eftpc):
    """
    Numerically stable Hill equation.
    """

    if pd.isna(ic50) or pd.isna(hill) or pd.isna(eftpc):
        return 0.0

    if ic50 <= 0 or hill <= 0 or eftpc <= 0:
        return 0.0

    try:
        ratio = (ic50 / eftpc) ** hill
        return 100.0 / (1.0 + ratio)

    except OverflowError:
        # If ratio is enormous, block is effectively 0%
        return 0.0
    
# ==========================================================
# Process All Drugs
# ==========================================================

rows = []

for index, drug in df.iterrows():

    print("\n" + "=" * 60)
    print(f"Processing {index + 1}/{len(df)} : {drug['Medication']}")
    print("=" * 60)

    blocks = {
        "IKr": hill_block(drug["IC50IKr"], drug["hIKr"], drug["EFTPC"]),
        "INa": hill_block(drug["IC50INa"], drug["hiNa"], drug["EFTPC"]),
        "INaL": hill_block(drug["IC50INaL"], drug["hINaL"], drug["EFTPC"]),
        "ICaL": hill_block(drug["IC50ICaL"], drug["hICaL"], drug["EFTPC"]),
        "IKs": hill_block(drug["IC50IKs"], drug["hIKs"], drug["EFTPC"]),
        "IK1": hill_block(drug["IC50IK1"], drug["hIK1"], drug["EFTPC"]),
        "Ito": hill_block(drug["IC50Ito"], drug["hIto"], drug["EFTPC"]),
    }

    simulator = ORDSimulator()

    simulator.apply_channel_blocks(
        ikr=blocks["IKr"],
        ina=blocks["INa"],
        inal=blocks["INaL"],
        ical=blocks["ICaL"],
        iks=blocks["IKs"],
        ik1=blocks["IK1"],
        ito=blocks["Ito"],
    )

    result = simulator.run()

    time = result["environment.time"]
    voltage = result["membrane.v"]

    extractor = APFeatureExtractor(time, voltage)
    features = extractor.extract_features()

    row = {
        "Medication": drug["Medication"],
        "Class": drug["Class"],
        "RMP": features["RMP"],
        "Peak": features["Peak"],
        "APD50": features["APD50"],
        "APD90": features["APD90"],
        "Triangulation": features["Triangulation"],
        "APA": features["Peak"] - features["RMP"],
        "Block_IKr": blocks["IKr"],
        "Block_INa": blocks["INa"],
        "Block_INaL": blocks["INaL"],
        "Block_ICaL": blocks["ICaL"],
        "Block_IKs": blocks["IKs"],
        "Block_IK1": blocks["IK1"],
        "Block_Ito": blocks["Ito"],
    }

    rows.append(row)

    print(f"✓ Completed {drug['Medication']}")


# ==========================================================
# Save Final Dataset
# ==========================================================

classifier_df = pd.DataFrame(rows)

classifier_df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print("All simulations completed!")
print("=" * 60)
print(f"Processed {len(classifier_df)} drugs")
print(f"Saved to: {OUTPUT_CSV}")