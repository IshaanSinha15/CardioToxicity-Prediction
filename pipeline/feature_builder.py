from typing import Dict
import pandas as pd

from classification_backend.feature_extraction.ap_features import APFeatureExtractor


FEATURE_ORDER = [
    "RMP",
    "Peak",
    "APD50",
    "APD90",
    "Triangulation",
    "APA",
    "Block_IKr",
    "Block_INa",
    "Block_INaL",
    "Block_ICaL",
    "Block_IKs",
    "Block_IK1",
    "Block_Ito",
]


def build_features_from_simulation(time, voltage, block_mapping: Dict[str, float]) -> Dict[str, float]:
    """Return a dict of features matching training order."""
    extractor = APFeatureExtractor(time, voltage)
    ap_features = extractor.extract_features()

    apa = ap_features["Peak"] - ap_features["RMP"]

    features = {
        "RMP": float(ap_features["RMP"]),
        "Peak": float(ap_features["Peak"]),
        "APD50": float(ap_features["APD50"]),
        "APD90": float(ap_features["APD90"]),
        "Triangulation": float(ap_features["Triangulation"]),
        "APA": float(apa),
        # Blocks: expected keys in block_mapping: IKr, INa, INaL, ICaL, IKs, IK1, Ito
        "Block_IKr": float(block_mapping.get("IKr", 0.0)),
        "Block_INa": float(block_mapping.get("INa", 0.0)),
        "Block_INaL": float(block_mapping.get("INaL", 0.0)),
        "Block_ICaL": float(block_mapping.get("ICaL", 0.0)),
        "Block_IKs": float(block_mapping.get("IKs", 0.0)),
        "Block_IK1": float(block_mapping.get("IK1", 0.0)),
        "Block_Ito": float(block_mapping.get("Ito", 0.0)),
    }

    # Ensure ordering
    ordered = {k: features[k] for k in FEATURE_ORDER}
    return ordered


def features_to_dataframe(features: Dict[str, float]) -> pd.DataFrame:
    """Convert dict to single-row DataFrame matching training columns."""
    df = pd.DataFrame([features])
    # Ensure column order
    df = df[FEATURE_ORDER]
    return df
