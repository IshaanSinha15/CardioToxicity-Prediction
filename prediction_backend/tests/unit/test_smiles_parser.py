import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
from prediction_backend.molecular_processing.smiles_parser import validate_smiles


def test_valid_smiles():

    mol = validate_smiles("CCO")
    assert mol is not None


def test_invalid_smiles():

    with pytest.raises(Exception):
        validate_smiles("INVALID_SMILES")