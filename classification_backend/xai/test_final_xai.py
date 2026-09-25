"""
test_final_xai.py

Tests the complete XAI pipeline.
"""

from .final_xai import FinalXAIPipeline


SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"

FEATURE_VECTOR = [
    -90.74,
    31.99,
    231.90,
    278.40,
    46.50,
    122.70,
    26.70,
    2.60,
    0.00,
    2.10,
    0.00,
    0.00,
    4.50,
    1000.0,
    5000.0,
    8000.0,
]


def main():

    pipeline = FinalXAIPipeline()

    result = pipeline.explain(
        smiles=SMILES,
        feature_vector=FEATURE_VECTOR,
    )

    print("\n========== FINAL XAI ==========\n")

    print(result)


if __name__ == "__main__":
    main()