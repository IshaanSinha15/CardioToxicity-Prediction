import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def clean_ic50(val):
    if pd.isnull(val): return np.nan
    s = str(val).strip().replace('>', '').replace('<', '').replace('=', '').replace(',', '')
    try:
        return float(s)
    except:
        return np.nan

def generate_comparison_plot(df_raw_pool, df_final, name):
    """Generates a side-by-side boxplot to show variance reduction."""
    plt.figure(figsize=(8, 6))

    # Prepare data for comparison
    raw_data = df_raw_pool[['ic50_nm']].copy()
    raw_data['Stage'] = 'Raw Data (TSV)'
    
    final_data = df_final[['ic50_nm']].copy()
    final_data['Stage'] = 'Final Training Set'
    
    combined = pd.concat([raw_data, final_data])

    # Plotting
    sns.boxplot(x='Stage', y='ic50_nm', data=combined, palette="Set1")
    plt.yscale('log')
    plt.title(f'{name}: Impact of Data Deduplication & Cleaning')
    plt.ylabel('IC50 (nM) - Log Scale')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'{name}_final_comparison.png')
    plt.show()

def process_ion_channel(raw_path, clean_path, output_name, channel_label):
    print(f"\n--- Processing {channel_label} ---")
    try:
        # 1. Load Raw (TSV) and Clean (Initial CSV)
        raw_df = pd.read_csv(raw_path, sep='\t' if raw_path.endswith('.tsv') else ',', low_memory=False)
        clean_df = pd.read_csv(clean_path)
        
        raw_df.columns = [c.lower().strip() for c in raw_df.columns]
        clean_df.columns = [c.lower().strip() for c in clean_df.columns]

        # 2. Extract Columns
        r_ic50 = [c for c in raw_df.columns if 'ic50' in c][0]
        r_smi = [c for c in raw_df.columns if 'smiles' in c][0]
        c_ic50 = [c for c in clean_df.columns if 'ic50' in c][0]
        c_smi = [c for c in clean_df.columns if 'smiles' in c][0]

        # 3. Standardize
        raw_df['ic50_nm'] = raw_df[r_ic50].apply(clean_ic50)
        clean_df['ic50_nm'] = clean_df[c_ic50].apply(clean_ic50)

        raw_sub = raw_df.dropna(subset=['ic50_nm'])[[r_smi, 'ic50_nm']].rename(columns={r_smi: 'smiles'})
        clean_sub = clean_df.dropna(subset=['ic50_nm'])[[c_smi, 'ic50_nm']].rename(columns={c_smi: 'smiles'})

        # 4. Define "Raw Pool" (The noise before averaging)
        all_raw_pool = pd.concat([raw_sub, clean_sub])

        # 5. Define "Final Set" (The clean averages)
        final_df = all_raw_pool.groupby('smiles')['ic50_nm'].mean().reset_index()

        # 6. Generate Single Comparison Graph
        generate_comparison_plot(all_raw_pool, final_df, channel_label)

        # 7. Save
        final_df.to_csv(f"{output_name}.csv", index=False)
        print(f"Saved {len(final_df)} unique rows.")

    except Exception as e:
        print(f"Error: {e}")

# --- Run Execution ---
datasets = [
    {"raw": "ishaansinha1501_gmail.com630.tsv", "clean": "hERG_IC50_clean.csv", "out": "hERG_final_training_unique", "label": "hERG"},
    {"raw": "ishaansinha1501_gmail.com629.tsv", "clean": "Nav1.5_IC50_clean.csv", "out": "nav1.5_final_training", "label": "Nav1.5"},
    {"raw": "cav12.tsv", "clean": "Cav1.2_IC50_clean.csv", "out": "cav1.2_final_training", "label": "Cav1.2"}
]

for ds in datasets:
    if os.path.exists(ds["raw"]) and os.path.exists(ds["clean"]):
        process_ion_channel(ds["raw"], ds["clean"], ds["out"], ds["label"])