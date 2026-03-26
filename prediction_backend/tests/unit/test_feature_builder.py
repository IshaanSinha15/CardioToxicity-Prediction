from data.feature_builder import build_features


def test_feature_builder():

    smiles = ["CCO", "CCC"]

    features = build_features(smiles)

    assert features.shape[0] == 2
    assert features.shape[1] > 2000   # sanity check