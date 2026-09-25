from .chemical_xai import chemical_xai

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

result = chemical_xai(smiles)

print("\n===== CHEMICAL XAI =====\n")

for explanation in result["explanation"]:

    print("Substructure :", explanation["substructure"])
    print("Importance   :", explanation["importance"])
    print("Explanation  :", explanation["message"])
    print("-" * 60)

from pathlib import Path

output = Path(__file__).parent / "results"
output.mkdir(exist_ok=True)

svg_path = output / "chemical_xai.svg"

with open(svg_path, "w", encoding="utf-8") as f:
    f.write(result["molecule_svg"])

print(f"\nSVG saved at:\n{svg_path.resolve()}")