import os
import time
import requests
import pandas as pd
from urllib.parse import quote


# ==========================================================
# Paths
# ==========================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "classification_backend",
    "dataset",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "validation_dataset.csv",
)


# ==========================================================
# Drug List
# ==========================================================

DRUGS = [

    "Amiodarone",
    "Ajmaline",
    "Astemizole",
    "Bepridil",
    "Cisapride",
    "Dofetilide",
    "Disopyramide",
    "Dronedarone",
    "Erythromycin",
    "Flecainide",
    "Haloperidol",
    "Ibutilide",
    "Lidocaine",
    "Mexiletine",
    "Moxifloxacin",
    "Nifedipine",
    "Nisoldipine",
    "Procainamide",
    "Quinidine",
    "Ranolazine",
    "Risperidone",
    "Sotalol",
    "Terfenadine",
    "Verapamil",
    "Ondansetron",
    "Chlorpromazine",
    "Clarithromycin",
    "Domperidone",
    "Droperidol",
    "Hydroxyzine",
    "Ziprasidone",
    "Clozapine",
    "Olanzapine",
    "Escitalopram",
    "Citalopram",
    "Sertraline",
    "Fluoxetine",
    "Paroxetine",
    "Venlafaxine",
    "Amitriptyline",
    "Imipramine",
    "Nortriptyline",
    "Carbamazepine",
    "Lamotrigine",
    "Phenytoin",
    "Valproate",
    "Levetiracetam",
    "Metoprolol",
    "Bisoprolol",
    "Propranolol",

]


# ==========================================================
# PubChem
# ==========================================================

PUBCHEM = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
    "compound/name/{}/property/CanonicalSMILES/JSON"
)


def get_smiles(drug):

    url = PUBCHEM.format(
        quote(drug)
    )

    try:

        r = requests.get(
            url,
            timeout=30,
        )

        if r.status_code != 200:
            return None

        data = r.json()

        return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]

    except Exception:

        return None


# ==========================================================
# Build Dataset
# ==========================================================

rows = []

print("=" * 60)
print("Downloading Drug Information")
print("=" * 60)

for i, drug in enumerate(DRUGS, start=1):

    print(f"[{i}/{len(DRUGS)}] {drug}")

    smiles = get_smiles(drug)

    rows.append({

        "Drug": drug,

        "SMILES": smiles,

        "Therapeutic_nM": None,

        "IC50_IKr": None,
        "IC50_INa": None,
        "IC50_ICaL": None,

        "Literature_Risk": None,

        "Reference": None,

    })

    time.sleep(0.2)   # stay within PubChem usage guidance

    # ==========================================================
# Create DataFrame
# ==========================================================

df = pd.DataFrame(rows)


# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 60)
print("Download Summary")
print("=" * 60)

print(f"Total Drugs : {len(df)}")

found = df["SMILES"].notna().sum()
missing = df["SMILES"].isna().sum()

print(f"SMILES Found   : {found}")
print(f"SMILES Missing : {missing}")


# ==========================================================
# Missing Drugs
# ==========================================================

if missing > 0:

    print("\nDrugs Not Found")

    print(
        df.loc[
            df["SMILES"].isna(),
            "Drug",
        ].tolist()
    )


# ==========================================================
# Remove Missing (Optional)
# ==========================================================

df = df.dropna(
    subset=["SMILES"]
).reset_index(drop=True)


# ==========================================================
# Remove Duplicate SMILES
# ==========================================================

duplicates = len(df)

df = df.drop_duplicates(
    subset="SMILES"
).reset_index(drop=True)

removed = duplicates - len(df)

print(f"\nDuplicate SMILES Removed : {removed}")


# ==========================================================
# Sort
# ==========================================================

df = df.sort_values(
    "Drug"
).reset_index(drop=True)


# ==========================================================
# Save
# ==========================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ==========================================================
# Preview
# ==========================================================

print("\nFirst 20 Entries")

print(df.head(20))


# ==========================================================
# Statistics
# ==========================================================

print("\nColumns")

print(df.columns.tolist())

print("\nMissing Values")

print(df.isnull().sum())

print("\nDataset Shape")

print(df.shape)


# ==========================================================
# Finished
# ==========================================================

print("\n" + "=" * 60)
print("Validation Dataset Created")
print("=" * 60)

print("\nSaved To")

print(OUTPUT_CSV)

print("\nDone.")