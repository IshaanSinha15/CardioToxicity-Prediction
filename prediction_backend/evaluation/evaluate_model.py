import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages

from prediction_backend.inference.predict import predict


DATASET = "data/datasets/final_combined.csv"
TASKS = ["herg", "nav", "cav"]

BASE_DIR = "evaluation"
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)


# -----------------------------
# Get scaffold
# -----------------------------
def get_scaffold(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


# -----------------------------
# Scaffold split
# -----------------------------
def scaffold_split(df, test_ratio=0.2):

    scaffolds = {}

    for idx, row in df.iterrows():

        scaffold = get_scaffold(row["smiles"])

        if scaffold is None:
            continue

        scaffolds.setdefault(scaffold, []).append(idx)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    train_idx = []
    test_idx = []

    total = len(df)
    test_target = int(total * test_ratio)

    for sset in scaffold_sets:

        if len(test_idx) < test_target:
            test_idx.extend(sset)
        else:
            train_idx.extend(sset)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    return train_df, test_df


# -----------------------------
# Main evaluation
# -----------------------------
def run_evaluation():

    print("\nLoading dataset...\n")

    df = pd.read_csv(DATASET)
    df.columns = [c.lower() for c in df.columns]

    train_df, test_df = scaffold_split(df)

    print(f"Train size: {len(train_df)}")
    print(f"Test size : {len(test_df)}")

    y_true = {t: [] for t in TASKS}
    y_pred = {t: [] for t in TASKS}

    total = len(test_df)

    print("\nRunning predictions...\n")

    for i, (_, row) in enumerate(test_df.iterrows()):

        smiles = row["smiles"]

        try:
            preds = predict(smiles)
        except Exception:
            continue

        for task in TASKS:

            if not np.isnan(row[task]):

                y_true[task].append(row[task])
                y_pred[task].append(preds[task])

        if i % 200 == 0:
            print(f"Processed {i}/{total}")

    print("\n========== Evaluation Results ==========\n")

    plot_files = []

    for task in TASKS:

        yt = np.array(y_true[task])
        yp = np.array(y_pred[task])

        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae = mean_absolute_error(yt, yp)
        r2 = r2_score(yt, yp)

        print(f"{task.upper()}")
        print(f"Samples : {len(yt)}")
        print(f"RMSE    : {rmse:.4f}")
        print(f"MAE     : {mae:.4f}")
        print(f"R2      : {r2:.4f}\n")

        # -----------------------------
        # Scatter Plot
        # -----------------------------
        plt.figure()

        plt.scatter(yt, yp, alpha=0.5)

        min_val = min(min(yt), min(yp))
        max_val = max(max(yt), max(yp))

        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        plt.xlabel("Actual pIC50")
        plt.ylabel("Predicted pIC50")
        plt.title(f"{task.upper()} Prediction")

        path = f"{PLOT_DIR}/{task}_scatter.png"
        plt.savefig(path)
        plot_files.append(path)
        plt.close()

        # -----------------------------
        # Residual Plot
        # -----------------------------
        residuals = yp - yt

        plt.figure()

        plt.scatter(yt, residuals, alpha=0.5)
        plt.axhline(0, linestyle="--")

        plt.xlabel("Actual pIC50")
        plt.ylabel("Residual (Predicted − Actual)")
        plt.title(f"{task.upper()} Residual Plot")

        path = f"{PLOT_DIR}/{task}_residual.png"
        plt.savefig(path)
        plot_files.append(path)
        plt.close()

        # -----------------------------
        # Applicability Domain
        # -----------------------------
        abs_residuals = np.abs(residuals)

        plt.figure()

        plt.scatter(yp, abs_residuals, alpha=0.5)

        threshold = np.percentile(abs_residuals, 95)
        plt.axhline(threshold, linestyle="--")

        plt.xlabel("Predicted pIC50")
        plt.ylabel("|Residual|")
        plt.title(f"{task.upper()} Applicability Domain")

        path = f"{PLOT_DIR}/{task}_applicability_domain.png"
        plt.savefig(path)
        plot_files.append(path)
        plt.close()

        # -----------------------------
        # Williams Plot
        # -----------------------------
        X = yp.reshape(-1,1)

        XtX_inv = np.linalg.inv(X.T @ X)
        H = X @ XtX_inv @ X.T
        leverage = np.diag(H)

        std_residuals = residuals / np.std(residuals)

        n = len(yt)
        p = X.shape[1]
        h_star = 3*(p+1)/n

        plt.figure()

        plt.scatter(leverage, std_residuals, alpha=0.5)

        plt.axhline(3, linestyle="--")
        plt.axhline(-3, linestyle="--")
        plt.axvline(h_star, linestyle="--")

        plt.xlabel("Leverage")
        plt.ylabel("Standardized Residuals")
        plt.title(f"{task.upper()} Williams Plot")

        path = f"{PLOT_DIR}/{task}_williams.png"
        plt.savefig(path)
        plot_files.append(path)
        plt.close()

    # -----------------------------
    # Create PDF report
    # -----------------------------
    pdf_path = os.path.join(BASE_DIR, "evaluation_report.pdf")

    with PdfPages(pdf_path) as pdf:

        for plot in plot_files:

            img = plt.imread(plot)

            fig = plt.figure(figsize=(8,6))
            plt.imshow(img)
            plt.axis("off")

            pdf.savefig(fig)
            plt.close()

    print("\nAll plots saved in: evaluation/plots/")
    print("PDF report saved as: evaluation/evaluation_report.pdf\n")


if __name__ == "__main__":
    run_evaluation()