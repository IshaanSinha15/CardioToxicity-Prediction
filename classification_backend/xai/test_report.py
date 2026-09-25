
from .chemical_xai import chemical_xai
from .report_generator import ReportGenerator

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

result = chemical_xai(smiles)

report = ReportGenerator().generate(result)

print(report)
