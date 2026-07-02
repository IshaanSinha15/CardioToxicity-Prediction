import os
import myokit
import matplotlib.pyplot as plt

from classification_backend.feature_extraction.ap_features import APFeatureExtractor


class ORDSimulator:
    def __init__(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        model_path = os.path.join(
            root_dir,
            "ord_engine",
            "cellml",
            "ToRORd_dynCl_epi.cellml"
        )

        importer = myokit.formats.importer("cellml")
        self.model = importer.model(model_path)

        print("Loaded model:", self.model.name())

    def apply_ikr_block(self, block_percent):
        """
        50 means reduce IKr conductance by 50%
        """
        gkr = self.model.get("IKr.GKr")

        original_rhs = gkr.rhs()
        scale = 1 - (block_percent / 100)

        gkr.set_rhs(f"({original_rhs}) * {scale}")

        print(f"Applied {block_percent}% IKr block")

    def run(self, duration=3000):
        sim = myokit.Simulation(self.model)

        # More sampling points for feature extraction
        data = sim.run(duration, log_interval=0.1)

        return data


if __name__ == "__main__":
    simulator = ORDSimulator()
    simulator.apply_ikr_block(50)

    result = simulator.run()

    time = result["environment.time"]
    voltage = result["membrane.v"]

    print("Simulation successful")
    print("Data points:", len(time))

    plt.figure(figsize=(10, 4))
    plt.plot(time, voltage)
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Voltage (mV)")
    plt.title("Action Potential")
    plt.grid(True)

    # Zoom only to first beat
    plt.xlim(0, 500)

    output_path = "outputs/action_potential.png"
    os.makedirs("outputs", exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    extractor = APFeatureExtractor(time, voltage)
    features = extractor.extract_features()

    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"{key}: {value:.2f}")