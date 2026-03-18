"""
predict.py — standalone prediction script (uses the same logic as app.py).

Usage:
    python predict.py "CC(=O)OC1=CC=CC=C1C(=O)O"
"""

import sys
import torch
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.DataStructs import ConvertToNumpyArray

from model_loader import load_model           # fixed architecture
from preprocessor  import smiles_to_features  # fixed: uses Morgan fp, not ASCII

# ── Load model ────────────────────────────────────────────────────────────────
model = load_model("model.pth")

# ── Diffusion schedule (must match training) ──────────────────────────────────
timesteps      = 1000
betas          = torch.linspace(0.0001, 0.02, timesteps)
alphas         = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
device         = torch.device("cpu")


@torch.no_grad()
def run_diffusion(seed_fp: np.ndarray) -> np.ndarray:
    """Partial forward + full reverse diffusion on a Morgan fingerprint."""
    x0      = torch.tensor(seed_fp, dtype=torch.float32).to(device)   # [1, 2048]
    t_start = 800
    noise   = torch.randn_like(x0)
    ac      = alphas_cumprod[t_start]
    x       = ac**0.5 * x0 + (1.0 - ac)**0.5 * noise

    for t in reversed(range(t_start)):
        t_tensor  = torch.tensor([t], dtype=torch.float32, device=device)
        pred      = model(x, t_tensor)
        alpha     = alphas[t]
        alpha_hat = alphas_cumprod[t]
        beta      = betas[t]
        noise_step = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = (1.0 / alpha**0.5) * (x - ((1.0 - alpha) / (1.0 - alpha_hat)**0.5) * pred) \
            + beta**0.5 * noise_step

    return x.squeeze(0).cpu().numpy()   # [2048]


def predict(smiles: str):
    """
    Given a SMILES string, run diffusion and return the generated fingerprint.
    Returns (generated_fp: np.ndarray shape [2048], probability_placeholder: float).
    """
    features = smiles_to_features(smiles)       # shape (1, 2048)
    gen_fp   = run_diffusion(features)          # shape (2048,)

    # Optional: report the mean activation as a rough "confidence"
    probability = float(np.mean(gen_fp > 0.5))
    return gen_fp, probability


if __name__ == "__main__":
    smiles = sys.argv[1] if len(sys.argv) > 1 else "CC(=O)OC1=CC=CC=C1C(=O)O"
    print(f"Input SMILES : {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("❌ Invalid SMILES string.")
        sys.exit(1)
    gen_fp, prob = predict(smiles)
    print(f"Generated FP : {gen_fp[:8]} ...  (2048 bits)")
    print(f"Bit density  : {prob:.3f}")