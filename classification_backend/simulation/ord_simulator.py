import os
import myokit
import matplotlib.pyplot as plt

from classification_backend.feature_extraction.ap_features import APFeatureExtractor


class ORDSimulator:
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
            "ToRORd_dynCl_epi.cellml"
        )

        importer = myokit.formats.importer("cellml")
        self.model = importer.model(model_path)

        print("Loaded model:", self.model.name())

    # ==========================================================
    # DEBUG
    # ==========================================================

    def inspect_component(self, component_name):
        """
        Print all variables inside a component.
        Useful only for debugging.
        """

        comp = self.model.get(component_name)

        print(f"\n========== {component_name} ==========")

        for var in comp.variables():
            print(var.name())

    # ==========================================================
    # GENERIC BLOCK FUNCTION
    # ==========================================================

    def _apply_block(self, variable_path, block_percent):
        """
        Reduce channel conductance/permeability by block_percent.
        """

        if block_percent <= 0:
            return

        try:
            var = self.model.get(variable_path)

            original_rhs = var.rhs()

            scale = 1 - (block_percent / 100.0)

            var.set_rhs(f"({original_rhs}) * {scale}")

            print(f"{variable_path} blocked by {block_percent:.2f}%")

        except Exception as e:
            print(f"Could not block {variable_path}")
            print(e)

    # ==========================================================
    # MULTI CHANNEL BLOCK
    # ==========================================================

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
        """
        Apply blocks to all supported ion channels.
        """

        channel_map = {

            # Rapid delayed rectifier
            "IKr.GKr": ikr,

            # Fast sodium
            "INa.GNa": ina,

            # Late sodium
            "INaL.GNaL": inal,

            # L-type calcium
            "ICaL.PCa": ical,

            # Slow delayed rectifier
            "IKs.GKs": iks,

            # Inward rectifier
            "IK1.GK1": ik1,

            # Transient outward
            "Ito.Gto": ito,
        }

        for variable, block in channel_map.items():
            self._apply_block(variable, block)

    # ==========================================================
    # BACKWARD COMPATIBILITY
    # ==========================================================

    def apply_ikr_block(self, block_percent):
        """
        Existing function kept so older code still works.
        """

        self.apply_channel_blocks(
            ikr=block_percent
        )

    # ==========================================================
    # RUN SIMULATION
    # ==========================================================

    def run(self, duration=3000):

        sim = myokit.Simulation(self.model)

        data = sim.run(
            duration,
            log_interval=0.1
        )

        return data


# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":

    simulator = ORDSimulator()

    # ----------------------------------------------------------
    # Uncomment ONLY if you want to inspect variables
    # ----------------------------------------------------------
    #
    # simulator.inspect_component("IKr")
    # simulator.inspect_component("INa")
    # simulator.inspect_component("INaL")
    # simulator.inspect_component("ICaL")
    # simulator.inspect_component("IKs")
    # simulator.inspect_component("IK1")
    # simulator.inspect_component("Ito")
    #

    # ----------------------------------------------------------
    # MULTI CHANNEL TEST
    # ----------------------------------------------------------

    simulator.apply_channel_blocks(
        ikr=50,
        ina=20,
        inal=10,
        ical=30,
        iks=15,
        ik1=10,
        ito=25
    )

    result = simulator.run()

    time = result["environment.time"]
    voltage = result["membrane.v"]

    print("\nSimulation successful")
    print("Data points:", len(time))

    plt.figure(figsize=(10, 4))

    plt.plot(time, voltage)

    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Voltage (mV)")
    plt.title("Action Potential")

    plt.grid(True)

    plt.xlim(0, 500)

    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/action_potential.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Plot saved to {output_path}")

    extractor = APFeatureExtractor(
        time,
        voltage
    )

    features = extractor.extract_features()

    print("\nExtracted Features:")

    for key, value in features.items():
        print(f"{key}: {value:.2f}")