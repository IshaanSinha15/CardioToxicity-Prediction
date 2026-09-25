"""
explanation.py

Generates human-readable explanations for
detected toxic substructures.
"""

MESSAGES = {

    "Aromatic Ring":
        "Aromatic rings are associated with hydrophobic interactions that can increase hERG channel binding.",

    "Phenol":
        "Phenol groups influence polarity and may affect ion-channel interactions.",

    "Tertiary Amine":
        "Tertiary amines are frequently present in compounds known to block the hERG channel.",

    "Piperidine":
        "Piperidine scaffolds are commonly observed in cardioactive compounds.",

    "Piperazine":
        "Piperazine rings may alter ion-channel affinity and pharmacokinetic properties.",

    "Imidazole":
        "Imidazole groups may influence receptor and ion-channel binding.",

    "Quinoline":
        "Quinoline derivatives have been associated with cardiotoxicity in several drugs."
}


def generate_explanation(substructures):
    explanations = []

    for item in substructures:

        explanations.append({
            "substructure": item["name"],
            "importance": item["importance"],
            "message": MESSAGES.get(
                item["name"],
                "No explanation available."
            )
        })

    return explanations