"""
preprocessor.py — converts a SMILES string to a 2048-bit Morgan fingerprint.

The old version used a 100-dimensional ASCII/length feature vector which is
a toy placeholder and is completely incompatible with the DiffusionModel that
expects 2048-dimensional inputs.  This version uses RDKit Morgan fingerprints,
matching the feature space used during training.
"""

import numpy as np
from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Shared generator — radius 2, 2048 bits (same as app.py and training)
_morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_features(smiles: str) -> np.ndarray:
    """
    Convert a SMILES string to a float32 numpy array of shape [1, 2048].

    Returns:
        np.ndarray of shape (1, 2048), dtype float32
    Raises:
        ValueError if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    fp  = _morgan_gen.GetFingerprint(mol)
    arr = np.zeros(2048, dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1)   # shape: (1, 2048)