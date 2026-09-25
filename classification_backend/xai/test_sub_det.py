from .chemical_xai import chemical_xai

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

result = chemical_xai(smiles)

print("\nChemical XAI Result\n")

for sub in result["substructures"]:
    print(sub)

print("\nExplanation:\n")

for exp in result["explanation"]:
    print(exp)