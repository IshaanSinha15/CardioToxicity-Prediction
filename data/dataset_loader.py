import pandas as pd
import numpy as np

def load_dataset(path):

    df = pd.read_csv(path)

    # normalize column names
    df.columns = [c.lower() for c in df.columns]

    # keep only needed columns
    df = df[["smiles", "ic50_nm"]]

    # remove missing values
    df = df.dropna()

    # force smiles to string
    df["smiles"] = df["smiles"].astype(str)

    # remove rows where smiles became numeric
    df = df[df["smiles"].str.contains("[A-Za-z]", regex=True)]

    # convert IC50 → pIC50
    df["pic50"] = 9 - np.log10(df["ic50_nm"])

    smiles = df["smiles"].tolist()
    labels = df["pic50"].values

    return smiles, labels