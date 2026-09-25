"""
report_generator.py
"""

class ReportGenerator:

    def generate(self, xai_result):

        return {
            "summary": {
                "num_substructures":
                    len(xai_result["substructures"])
            },

            "substructures":
                xai_result["substructures"],

            "explanation":
                xai_result["explanation"],

            "molecule_svg":
                xai_result["molecule_svg"]
        }