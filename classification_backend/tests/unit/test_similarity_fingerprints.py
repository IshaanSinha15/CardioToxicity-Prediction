import pytest

from classification_backend.assessment.similarity.fingerprints import (
    FingerprintConfig,
    FingerprintError,
    canonicalize_smiles,
    morgan_fingerprint,
    parse_smiles,
)


def test_fingerprint_is_deterministic_and_configured():
    config = FingerprintConfig()
    molecule = parse_smiles("C(C)O")

    first = morgan_fingerprint(molecule, config)
    second = morgan_fingerprint(molecule, config)

    assert first.ToBitString() == second.ToBitString()
    assert first.GetNumBits() == 2048
    assert config.method() == "Morgan radius 2, 2048 bits, Tanimoto"


def test_canonicalization_handles_equivalent_and_stereo_smiles():
    assert canonicalize_smiles(parse_smiles("C(C)O")) == canonicalize_smiles(parse_smiles("CCO"))
    assert canonicalize_smiles(parse_smiles("C[C@H](O)F")) != canonicalize_smiles(parse_smiles("C[C@@H](O)F"))


@pytest.mark.parametrize("smiles", ["", "not a smiles", None, 12])
def test_invalid_smiles_are_rejected(smiles):
    with pytest.raises(FingerprintError):
        parse_smiles(smiles)