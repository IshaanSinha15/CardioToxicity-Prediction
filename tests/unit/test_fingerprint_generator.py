import numpy as np
import pytest
from molecular_processing.fingerprint_generator import generate_fingerprint


def test_fingerprint_size():
    fp = generate_fingerprint("CCO")
    assert len(fp) == 2048


def test_fingerprint_type():
    fp = generate_fingerprint("CCO")
    assert isinstance(fp, np.ndarray)


def test_invalid_smiles():
    with pytest.raises(ValueError):
        generate_fingerprint("INVALID_SMILES")