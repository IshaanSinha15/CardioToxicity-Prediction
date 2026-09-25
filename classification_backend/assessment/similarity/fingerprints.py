from __future__ import annotations

from dataclasses import asdict, dataclass

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class FingerprintError(ValueError):
    pass


@dataclass(frozen=True)
class FingerprintConfig:
    family: str = "Morgan"
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = True

    def __post_init__(self) -> None:
        if self.family != "Morgan":
            raise FingerprintError("only Morgan fingerprints are supported")
        if self.radius < 0 or self.n_bits <= 0:
            raise FingerprintError("radius must be non-negative and n_bits must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def method(self) -> str:
        return f"{self.family} radius {self.radius}, {self.n_bits} bits, Tanimoto"


def parse_smiles(smiles: object) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        raise FingerprintError("SMILES must be a non-empty string")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise FingerprintError("SMILES could not be parsed")
    return molecule


def canonicalize_smiles(molecule: Chem.Mol) -> str:
    if molecule is None:
        raise FingerprintError("molecule cannot be None")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def morgan_fingerprint(molecule: Chem.Mol, config: FingerprintConfig | None = None):
    config = config or FingerprintConfig()
    if molecule is None:
        raise FingerprintError("molecule cannot be None")
    try:
        generator = AllChem.GetMorganGenerator(
            radius=config.radius,
            fpSize=config.n_bits,
            includeChirality=config.use_chirality,
        )
        return generator.GetFingerprint(molecule)
    except AttributeError:
        return AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            config.radius,
            nBits=config.n_bits,
            useChirality=config.use_chirality,
        )


def tanimoto_similarity(first, second) -> float:
    return float(DataStructs.TanimotoSimilarity(first, second))