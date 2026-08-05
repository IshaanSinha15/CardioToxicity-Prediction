"""Simple runner for the prediction pipeline.

Usage:
    python -m pipeline.run_pipeline --smiles "CCO" --dose 100
"""
import argparse
import json

from .prediction_pipeline import PredictionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=False)
    parser.add_argument("--dose", required=False, type=float)
    parser.add_argument("--skip-sim", action="store_true", help="Skip ORd simulation if myokit is unavailable")
    parser.add_argument("--drug-name", required=False)
    parser.add_argument("--interactive", action="store_true", help="Run interactive prompt to type SMILES and dose")
    args = parser.parse_args()

    pipeline = PredictionPipeline()

    # CLI single-run mode
    if not args.interactive and args.smiles:
        payload = {"smiles": args.smiles, "dose_nm": args.dose, "drug_name": args.drug_name}
        if args.skip_sim:
            payload["skip_simulation"] = True

        result = pipeline.run(payload)
        print(json.dumps(result, indent=2))
        return

    # Interactive loop mode
    print("\nInteractive prediction mode. Type 'exit' to quit.")
    while True:
        smiles = input("Enter SMILES (or type 'exit'): ").strip()
        if smiles.lower() == "exit":
            break

        dose_val = input("Enter concentration in nM (or type 'exit'): ").strip()
        if dose_val.lower() == "exit":
            break

        try:
            dose = float(dose_val)
        except ValueError:
            print("Please enter a valid numeric concentration.\n")
            continue

        # Allow per-run drug name (falls back to CLI-provided name)
        drug_name = input(f"Enter drug name (optional) [{args.drug_name or ''}]: ").strip()
        if drug_name == "":
            drug_name = args.drug_name

        # Allow per-run override of skip-sim
        use_skip = args.skip_sim
        skip_input = input(f"Skip simulation? (y/N) [{'Y' if args.skip_sim else 'N'}]: ").strip().lower()
        if skip_input in ("y", "yes"):
            use_skip = True
        elif skip_input in ("n", "no"):
            use_skip = False

        payload = {"smiles": smiles, "dose_nm": dose, "drug_name": drug_name}
        if use_skip:
            payload["skip_simulation"] = True

        try:
            result = pipeline.run(payload)
        except Exception as e:
            print("Prediction failed:", e)
            continue

        # Print JSON for easy consumption by web clients later
        print(json.dumps(result, indent=2))



if __name__ == "__main__":
    main()
