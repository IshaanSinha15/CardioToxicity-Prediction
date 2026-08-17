import os
import myokit
if __name__ == "__main__":
    import matplotlib.pyplot as plt

from classification_backend.feature_extraction.ap_features import (
    APFeatureExtractor,
)


class ORDSimulator:
    """
    Optimized ORd simulator.

    Improvements:
    ----------------
    • CellML compiled only once.
    • One Simulation object reused.
    • No set_rhs().
    • No recompilation.
    • Conductances changed using set_constant().
    • Simulation reset before every run.
    """

    def __init__(self):

        root_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__)
            )
        )

        model_path = os.path.join(
            root_dir,
            "ord_engine",
            "cellml",
            "ToRORd_dynCl_epi.cellml",
        )

        importer = myokit.formats.importer("cellml")
        self.model = importer.model(model_path)

        # Compile ONCE
        # Create pacing protocol
        protocol = myokit.pacing.blocktrain(
            period=1000,
            duration=0.5,
            offset=0,
            level=1,
        )

        # Compile once
        self.sim = myokit.Simulation(
            self.model,
            protocol,
        )

        # ----------------------------------------
        # Variables that should be modified
        # ----------------------------------------

        self.constant_map = {
            "ina": "INa.GNa",
            "inal": "INaL.GNaL_b",
            "ical": "ICaL.PCa_b",
            "ikr": "IKr.GKr_b",
            "iks": "IKs.GKs_b",
            "ik1": "IK1.GK1_b",
            "ito": "Ito.Gto_b",
        }

        # ----------------------------------------
        # Store original values once
        # ----------------------------------------

        self.baseline = {}

        for key, var_name in self.constant_map.items():
            var = self.model.get(var_name)

            # Get the constant's current value from its RHS
            value = var.eval()

            self.baseline[key] = float(value)

        # Requested blocks (%)
        self.blocks = {
            "ina": 0.0,
            "inal": 0.0,
            "ical": 0.0,
            "ikr": 0.0,
            "iks": 0.0,
            "ik1": 0.0,
            "ito": 0.0,
        }

    # ======================================================
    # Optional debugging helper
    # ======================================================

    def inspect_component(self, component_name):

        comp = self.model.get(component_name)

        print(f"\n========== {component_name} ==========")

        for var in comp.variables():
            print(var.name())

    # ======================================================
    # Store requested channel blocks
    # (No recompilation here)
    # ======================================================

    def apply_channel_blocks(
        self,
        ikr=0,
        ina=0,
        inal=0,
        ical=0,
        iks=0,
        ik1=0,
        ito=0,
    ):

        self.blocks["ikr"] = float(ikr)
        self.blocks["ina"] = float(ina)
        self.blocks["inal"] = float(inal)
        self.blocks["ical"] = float(ical)
        self.blocks["iks"] = float(iks)
        self.blocks["ik1"] = float(ik1)
        self.blocks["ito"] = float(ito)

    # Backward compatibility
    def apply_ikr_block(self, block_percent):
        self.apply_channel_blocks(ikr=block_percent)

    # ======================================================
    # Restore baseline constants
    # ======================================================

    def _restore_constants(self):

        for key, variable in self.constant_map.items():
            self.sim.set_constant(
                variable,
                self.baseline[key]
            )

    # ======================================================
    # Apply current block percentages
    # ======================================================

    def _apply_constants(self):

        for key, variable in self.constant_map.items():

            original = self.baseline[key]

            block = max(0.0, min(100.0, self.blocks[key]))

            scale = 1.0 - (block / 100.0)

            self.sim.set_constant(
                variable,
                float(original * scale)
            )

    # ======================================================
    # Run Simulation
    # ======================================================

    def run(
        self,
        duration=3000,
        prepace=1000,
        log_interval=0.1,
    ):

        self.sim.reset()

        self._restore_constants()
        self._apply_constants()

        if prepace > 0:
            self.sim.pre(prepace)

        data = self.sim.run(
            duration,
            log_interval=log_interval,
        )

        return data


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    simulator = ORDSimulator()

    simulator.apply_channel_blocks(
        ikr=50,
        ina=20,
        inal=10,
        ical=30,
        iks=15,
        ik1=10,
        ito=25,
    )

    result = simulator.run()

    time = result["environment.time"]
    voltage = result["membrane.v"]

    print("Simulation successful")
    print("Points:", len(time))

    plt.figure(figsize=(10, 4))
    plt.plot(time, voltage)

    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Action Potential")

    plt.grid(True)
    plt.xlim(0, 500)

    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/action_potential.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("Saved:", output_path)

    extractor = APFeatureExtractor(
        time,
        voltage,
    )

    features = extractor.extract_features()

    print("\nExtracted Features")

    for key, value in features.items():
        print(f"{key}: {value:.2f}")