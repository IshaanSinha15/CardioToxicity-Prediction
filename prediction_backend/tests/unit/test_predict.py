import pandas as pd
from pathlib import Path
from prediction_backend.inference.predict import predict

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET = REPO_ROOT / "data" / "datasets" / "final_combined.csv"
TARGET_MEAN_PATH = REPO_ROOT / "data" / "target_mean.csv"
TARGET_STD_PATH = REPO_ROOT / "data" / "target_std.csv"
TASKS = ["herg", "nav", "cav"]


def load_target_stats():

    mean = pd.read_csv(TARGET_MEAN_PATH, index_col=0).iloc[:, 0].to_dict()
    std = pd.read_csv(TARGET_STD_PATH, index_col=0).iloc[:, 0].to_dict()

    return mean, std


def find_actual(smiles, df, mean, std):

    row = df[df["smiles"] == smiles]

    if len(row) == 0:
        return None

    row = row.iloc[0]

    actual = {}

    for task in TASKS:

        z_val = row[task]

        if pd.isna(z_val):
            actual[task] = float("nan")
        else:
            actual[task] = z_val * std[task] + mean[task]

    return actual


def run_test():

    print("\n===== Prediction Test =====\n")

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")
    if not TARGET_MEAN_PATH.exists():
        raise FileNotFoundError(f"Target mean file not found: {TARGET_MEAN_PATH}")
    if not TARGET_STD_PATH.exists():
        raise FileNotFoundError(f"Target std file not found: {TARGET_STD_PATH}")

    df = pd.read_csv(DATASET)
    df.columns = [c.lower() for c in df.columns]
    target_mean, target_std = load_target_stats()

    while True:

        smiles = input("Enter SMILES (or type 'exit'): ")

        if smiles.lower() == "exit":
            break

        try:
            preds = predict(smiles)
        except Exception as e:
            print("Prediction failed:", e)
            continue

        actual = find_actual(smiles, df, target_mean, target_std)

        print("\nPredicted Values\n")

        for task, values in preds.items():

            print(task.upper())
            print(f"pIC50   : {values['pIC50']:.4f}")
            print(f"IC50 nM : {values['IC50_nM']:.2e}\n")

        if actual is not None:

            print("Actual Values\n")

            for task in TASKS:

                val = actual[task]

                if pd.isna(val):
                    print(f"{task.upper()} pIC50 : NaN")
                else:
                    print(f"{task.upper()} pIC50 : {val:.4f}")

        else:
            print("\n Compound not in dataset")

        if actual is not None:

            print("\nPrediction Error\n")

            for task in TASKS:

                actual_val = actual[task]

                if pd.isna(actual_val):
                    continue

                pred_val = preds[task]["pIC50"]

                error = abs(pred_val - actual_val)

                print(f"{task.upper()} error : {error:.4f}")

        print("\n-----------------------------\n")


if __name__ == "__main__":
    run_test()