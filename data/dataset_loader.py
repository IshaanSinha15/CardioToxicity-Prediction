import pandas as pd


def load_dataset(path):

    df = pd.read_csv(path)

    # remove missing values
    df = df.dropna()

    smiles = df["smiles"].values
    labels = df["IC50"].values

    return smiles, labels