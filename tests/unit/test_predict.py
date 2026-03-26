import pandas as pd
from inference.predict import predict

DATASET = "data/datasets/final_combined.csv"


def find_actual(smiles, df):

    row = df[df["smiles"] == smiles]

    if len(row) == 0:
        return None

    row = row.iloc[0]

    return {
        "herg": row["herg"],
        "nav": row["nav"],
        "cav": row["cav"]
    }


def run_test():

    print("\n===== Prediction Test =====\n")

    df = pd.read_csv(DATASET)
    df.columns = [c.lower() for c in df.columns]

    while True:

        smiles = input("Enter SMILES (or type 'exit'): ")

        if smiles.lower() == "exit":
            break

        try:
            preds = predict(smiles)
        except Exception as e:
            print("Prediction failed:", e)
            continue

        actual = find_actual(smiles, df)

        print("\nPredicted Values\n")

        for task, values in preds.items():

            print(task.upper())
            print(f"pIC50   : {values['pIC50']:.4f}")
            print(f"IC50 nM : {values['IC50_nM']:.2e}\n")

        if actual is not None:

            print("Actual Values\n")

            for task in ["herg", "nav", "cav"]:

                val = actual[task]

                if pd.isna(val):
                    print(f"{task.upper()} pIC50 : NaN")
                else:
                    print(f"{task.upper()} pIC50 : {val:.4f}")

        else:
            print("\nActual values not found in dataset")

        if actual is not None:

            print("\nPrediction Error\n")

            for task in ["herg", "nav", "cav"]:

                actual_val = actual[task]

                if pd.isna(actual_val):
                    continue

                pred_val = preds[task]["pIC50"]

                error = abs(pred_val - actual_val)

                print(f"{task.upper()} error : {error:.4f}")

        print("\n-----------------------------\n")


if __name__ == "__main__":
    run_test()