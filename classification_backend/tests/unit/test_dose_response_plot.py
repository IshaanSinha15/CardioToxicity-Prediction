import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Hill Equation
# -----------------------------
def calculate_block(ic50, concentration, hill_coefficient=1):
    """
    Calculate channel block using Hill equation

    Block = C^h / (IC50^h + C^h)
    """
    numerator = concentration ** hill_coefficient
    denominator = (ic50 ** hill_coefficient) + numerator
    return numerator / denominator


# -----------------------------
# Concentration Sweep
# -----------------------------
def generate_concentrations():
    """
    Generate concentration sweep in nM
    """
    return np.array([1, 10, 100, 1000, 10000, 100000])


# -----------------------------
# Plotting
# -----------------------------
def plot_channel_curve(concentrations, blocks, channel_name, save_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(concentrations, blocks, marker="o")
    plt.xscale("log")
    plt.xlabel("Concentration (nM)")
    plt.ylabel("Block Fraction")
    plt.title(f"{channel_name} Dose Response")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{channel_name}_dose_response.png"))
    plt.close()


def plot_combined(concentrations, herg, nav, cav, save_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(concentrations, herg, marker='o', label='hERG')
    plt.plot(concentrations, nav, marker='o', label='Nav1.5')
    plt.plot(concentrations, cav, marker='o', label='Cav1.2')

    plt.xscale("log")
    plt.xlabel("Concentration (nM)")
    plt.ylabel("Block Fraction")
    plt.title("Combined Dose Response")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "combined_dose_response.png"))
    plt.close()


# -----------------------------
# Main Test Script
# -----------------------------
def main():
    save_dir = "outputs/dose_response_plots"
    os.makedirs(save_dir, exist_ok=True)

    # Sample IC50 values from Ishaan's output
    herg_ic50 = 2330
    nav_ic50 = 12400
    cav_ic50 = 54400

    concentrations = generate_concentrations()

    herg_blocks = []
    nav_blocks = []
    cav_blocks = []

    for c in concentrations:
        herg_blocks.append(calculate_block(herg_ic50, c))
        nav_blocks.append(calculate_block(nav_ic50, c))
        cav_blocks.append(calculate_block(cav_ic50, c))

    # Print Table
    df = pd.DataFrame({
        "Concentration (nM)": concentrations,
        "hERG Block %": np.array(herg_blocks) * 100,
        "Nav1.5 Block %": np.array(nav_blocks) * 100,
        "Cav1.2 Block %": np.array(cav_blocks) * 100,
    })

    print("\nDose Response Table")
    print(df.to_string(index=False))

    # Individual plots
    plot_channel_curve(concentrations, herg_blocks, "hERG", save_dir)
    plot_channel_curve(concentrations, nav_blocks, "Nav1.5", save_dir)
    plot_channel_curve(concentrations, cav_blocks, "Cav1.2", save_dir)

    # Combined plot
    plot_combined(
        concentrations,
        herg_blocks,
        nav_blocks,
        cav_blocks,
        save_dir
    )

    print(f"\nPlots saved in: {save_dir}")


if __name__ == "__main__":
    main()