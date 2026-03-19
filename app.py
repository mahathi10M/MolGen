import warnings, os, math
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, AllChem
from rdkit.Chem.rdmolops import AddHs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app, origins=[
    "https://your-netlify-url.netlify.app",
    "https://mahathi10m.github.io",
    "http://localhost:5173",
    "http://localhost:3000"
])
# ── Model (sinusoidal time embedding — must match training architecture) ───────
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args  = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class DiffusionModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),            nn.SiLU(), nn.LayerNorm(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),       nn.SiLU(), nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, input_dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, self.time_embed(t)], dim=-1))

device = torch.device("cpu")
model  = DiffusionModel().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
print("✅ Model loaded")

# ── Morgan fingerprint generator ──────────────────────────────────────────────
_morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

# ── Diffusion schedule (keep everything on CPU; move to device if using GPU) ──
timesteps      = 1000
betas          = torch.linspace(0.0001, 0.02, timesteps)   # shape [T]
alphas         = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)               # shape [T]

# ── Training data ─────────────────────────────────────────────────────────────
CSV_PATH = "SMILES_Big_Data_Set.csv"
training_fps, training_smiles = None, None
# Pre-built RDKit fingerprint objects for fast Tanimoto lookup
_rdkit_fps = None

def load_training_data():
    global training_fps, training_smiles, _rdkit_fps
    if not os.path.exists(CSV_PATH):
        print(f"⚠️  {CSV_PATH} not found — nearest-SMILES lookup disabled")
        return
    print("📂 Loading training SMILES...")
    df = pd.read_csv(CSV_PATH)
    smiles_list = df['SMILES'].dropna().unique()[:5000]
    fps, valid, rdkit_fps = [], [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = _morgan_gen.GetFingerprint(mol)
            arr = np.zeros(2048, dtype=np.uint8)
            # Convert ExplicitBitVect → numpy array
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid.append(s)
            rdkit_fps.append(fp)          # keep native RDKit fp for fast Tanimoto
    training_fps    = np.array(fps)       # shape [N, 2048], uint8
    training_smiles = valid
    _rdkit_fps      = rdkit_fps           # list of ExplicitBitVect
    print(f"✅ Loaded {len(training_smiles)} training molecules")

load_training_data()

# ── Diffusion sampler ─────────────────────────────────────────────────────────
@torch.no_grad()
def run_diffusion(seed_fp: np.ndarray) -> np.ndarray:
    """
    Partial forward-then-reverse diffusion.
    seed_fp: float32 numpy array of shape [2048]
    Returns: float32 numpy array of shape [2048]
    """
    x0      = torch.tensor(seed_fp, dtype=torch.float32).unsqueeze(0).to(device)
    t_start = 800

    # Forward: add noise up to t_start
    noise = torch.randn_like(x0)
    ac    = alphas_cumprod[t_start].to(device)
    x     = ac**0.5 * x0 + (1.0 - ac)**0.5 * noise

    # Reverse: denoise from t_start → 0
    for t in reversed(range(t_start)):
        t_tensor   = torch.tensor([t], dtype=torch.float32, device=device)
        pred       = model(x, t_tensor)

        alpha      = alphas[t].to(device)
        alpha_hat  = alphas_cumprod[t].to(device)
        beta       = betas[t].to(device)

        noise_step = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = (1.0 / alpha**0.5) * (x - ((1.0 - alpha) / (1.0 - alpha_hat)**0.5) * pred) \
            + beta**0.5 * noise_step

    return x.squeeze(0).cpu().numpy()

# ── Nearest SMILES (vectorised Tanimoto via RDKit bulk similarity) ────────────
def nearest_smiles(gen_fp: np.ndarray):
    """
    Finds the most similar training SMILES to the generated fingerprint.
    Uses RDKit BulkTanimotoSimilarity for speed (no Python loop over molecules).
    """
    if _rdkit_fps is None:
        return None

    # Convert continuous float fp → binary string → RDKit ExplicitBitVect
    bin_str = ''.join('1' if b > 0.5 else '0' for b in gen_fp)
    try:
        gen_rdkit = DataStructs.CreateFromBitString(bin_str)
    except Exception:
        return None

    # Bulk Tanimoto: returns a list of floats, length = len(_rdkit_fps)
    sims     = DataStructs.BulkTanimotoSimilarity(gen_rdkit, _rdkit_fps)
    best_idx = int(np.argmax(sims))
    return training_smiles[best_idx]

# ── SMILES → 2D PNG (base64) ──────────────────────────────────────────────────
def smiles_to_image(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    img = Draw.MolToImage(mol, size=(400, 400))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── SMILES → 3D SDF ───────────────────────────────────────────────────────────
def smiles_to_3d_sdf(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    try:
        mol = AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result == -1:
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result == -1:
            return None
        AllChem.MMFFOptimizeMolecule(mol)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None

# ── Lipinski Ro5 properties ───────────────────────────────────────────────────
def lipinski_check(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"valid": False}
    mw   = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    return {
        "valid": True,
        "molecular_weight": mw,
        "logP": logp,
        "h_bond_donors": hbd,
        "h_bond_acceptors": hba,
        "drug_like": (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10),
    }

# ── SMILES → float32 Morgan fingerprint ──────────────────────────────────────
def smiles_to_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    from rdkit.DataStructs import ConvertToNumpyArray
    fp  = _morgan_gen.GetFingerprint(mol)
    arr = np.zeros(2048, dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr

# ── /predict endpoint ─────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data   = request.get_json()
    smiles = data.get("smiles", "").strip()

    if not smiles:
        return jsonify({"error": "SMILES not provided"}), 400
    if not Chem.MolFromSmiles(smiles):
        return jsonify({"error": "Invalid SMILES. Try: CC(=O)OC1=CC=CC=C1C(=O)O"}), 400

    try:
        input_image = smiles_to_image(smiles)
        input_props = lipinski_check(smiles)
        input_sdf   = smiles_to_3d_sdf(smiles)

        seed_fp    = smiles_to_fp(smiles)
        if seed_fp is None:
            return jsonify({"error": "Could not compute fingerprint for input SMILES"}), 400

        gen_fp     = run_diffusion(seed_fp)
        gen_smiles = nearest_smiles(gen_fp) if training_smiles else smiles
        gen_image  = smiles_to_image(gen_smiles) if gen_smiles else None
        gen_props  = lipinski_check(gen_smiles)  if gen_smiles else None
        gen_sdf    = smiles_to_3d_sdf(gen_smiles) if gen_smiles else None

        return jsonify({
            "input": {
                "smiles": smiles,
                "image":  input_image,
                "properties": input_props,
                "sdf":    input_sdf,
            },
            "generated": {
                "smiles": gen_smiles,
                "image":  gen_image,
                "properties": gen_props,
                "sdf":    gen_sdf,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── /health endpoint ──────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "molecules": len(training_smiles) if training_smiles else 0,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
