import warnings, os, math, json, re

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# ── Silence ALL RDKit stderr (kekulize, valence, parse errors) ─────────────────
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem, QED
from rdkit.Chem import SanitizeMol
from rdkit.Chem.rdmolops import AddHs
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app, origins="*")

device = torch.device("cpu")

# ── 1. Load tokenizer from checkpoint ─────────────────────────────────────────
ckpt = torch.load("model_v2.pth", map_location=device, weights_only=False)

token2idx = ckpt['token2idx']
idx2token = {int(i): t for t, i in token2idx.items()}
VOCAB_SIZE = ckpt['vocab_size']
PAD_IDX    = ckpt['pad_idx']
SOS_IDX    = ckpt['sos_idx']
EOS_IDX    = ckpt['eos_idx']
MAX_LEN    = 85   # reduced from 120 — cuts garbage long-tail sequences

print(f"✅ Tokenizer loaded | vocab={VOCAB_SIZE} PAD={PAD_IDX} SOS={SOS_IDX} EOS={EOS_IDX}")

# ── Pre-compute constraint index sets from vocab ───────────────────────────────
UNK_IDX   = token2idx.get('<UNK>', -1)
OPEN_IDX  = token2idx.get('(', -1)
CLOSE_IDX = token2idx.get(')', -1)

BOND_TOKENS = {'#', '=', '-'}
BOND_IDXS   = frozenset(token2idx[t] for t in BOND_TOKENS if t in token2idx)

DIGIT_IDXS = {token2idx[d]: d for d in '1234567' if d in token2idx}

AROMATIC_ATOM_TOKENS = {'c', 'n', 'o', 's', '[nH]', '[n+]', '[o+]', '[se]'}
AROMATIC_IDXS = frozenset(token2idx[t] for t in AROMATIC_ATOM_TOKENS if t in token2idx)

# ── 2. Model — exact match to notebook Cell 6 ─────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MolTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_layers=6, dim_ff=1024, dropout=0.1, max_len=122):
        super().__init__()
        self.d_model   = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc   = PositionalEncoding(d_model, dropout, max_len + 10)
        decoder_layer  = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output      = nn.Linear(d_model, vocab_size)
        self.register_buffer('_mem_dummy', torch.zeros(1, 1, d_model))

    def forward(self, x):
        B, T  = x.shape
        emb   = self.pos_enc(self.embedding(x) * math.sqrt(self.d_model))
        mask  = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        pad_m = (x == PAD_IDX).bool()   # explicit bool — kills UserWarning
        mem   = self._mem_dummy.expand(B, -1, -1)
        out   = self.transformer(emb, mem,
                                 tgt_mask=mask,
                                 tgt_key_padding_mask=pad_m)
        return self.output(out)

    @torch.no_grad()
    def generate(self, n=10, temperature=1.0, top_k=10):
        self.eval()
        ids  = torch.full((n, 1), SOS_IDX, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)

        open_parens  = [0] * n
        open_rings   = [set() for _ in range(n)]
        pending_bond = [False] * n
        in_aromatic  = [False] * n

        for step in range(MAX_LEN):
            logits = self.forward(ids)[:, -1, :] / max(temperature, 0.1)

            # Block globally invalid tokens
            logits[:, PAD_IDX] = float('-inf')
            logits[:, SOS_IDX] = float('-inf')
            if UNK_IDX >= 0:
                logits[:, UNK_IDX] = float('-inf')

            tokens_left = MAX_LEN - step

            for i in range(n):
                if done[i]:
                    continue

                # Error 3: no ) without open paren
                if open_parens[i] == 0 and CLOSE_IDX >= 0:
                    logits[i, CLOSE_IDX] = float('-inf')

                # Error 3: no EOS while structure is open
                if open_parens[i] > 0 or len(open_rings[i]) > 0:
                    logits[i, EOS_IDX] = float('-inf')

                # Error 4: after a bond, only atoms are valid next
                if pending_bond[i]:
                    for bidx in BOND_IDXS:
                        logits[i, bidx] = float('-inf')
                    if CLOSE_IDX >= 0:
                        logits[i, CLOSE_IDX] = float('-inf')
                    logits[i, EOS_IDX] = float('-inf')
                    if OPEN_IDX >= 0:
                        logits[i, OPEN_IDX] = float('-inf')

                # Error 1: block aromatic atom tokens outside aromatic context
                if not in_aromatic[i]:
                    for aidx in AROMATIC_IDXS:
                        logits[i, aidx] = float('-inf')

                # Error 3: cap open rings at 4 to prevent insane polycyclics
                for didx, dtok in DIGIT_IDXS.items():
                    if len(open_rings[i]) >= 4 and dtok not in open_rings[i]:
                        logits[i, didx] = float('-inf')

                # Error 3: near max_len — block opening any new structure
                if tokens_left <= 8:
                    if OPEN_IDX >= 0:
                        logits[i, OPEN_IDX] = float('-inf')
                    for bidx in BOND_IDXS:
                        logits[i, bidx] = float('-inf')
                    for didx, dtok in DIGIT_IDXS.items():
                        if dtok not in open_rings[i]:
                            logits[i, didx] = float('-inf')

                # Error 3: final token — force closing tokens only
                if tokens_left <= 1:
                    allow = {EOS_IDX}
                    if open_parens[i] > 0 and CLOSE_IDX >= 0:
                        allow.add(CLOSE_IDX)
                    for didx, dtok in DIGIT_IDXS.items():
                        if dtok in open_rings[i]:
                            allow.add(didx)
                    block_mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=device)
                    for idx in allow:
                        block_mask[idx] = False
                    logits[i] = logits[i].masked_fill(block_mask, float('-inf'))

            if top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits  = logits.masked_fill(logits < vals[:, -1:], float('-inf'))

            probs   = torch.softmax(logits, dim=-1)
            probs   = torch.nan_to_num(probs, nan=1.0 / probs.size(-1))
            probs   = probs / probs.sum(dim=-1, keepdim=True)
            nxt     = torch.multinomial(probs, 1).squeeze(1)

            for i in range(n):
                if done[i]:
                    continue
                tok = idx2token.get(nxt[i].item(), '')

                if tok == '(':
                    open_parens[i] += 1
                    pending_bond[i] = False
                elif tok == ')':
                    open_parens[i] = max(0, open_parens[i] - 1)
                    pending_bond[i] = False
                elif tok in BOND_TOKENS:
                    pending_bond[i] = True
                elif tok in DIGIT_IDXS.values():
                    pending_bond[i] = False
                    if tok in open_rings[i]:
                        open_rings[i].discard(tok)
                    else:
                        open_rings[i].add(tok)
                else:
                    pending_bond[i] = False
                    if tok in AROMATIC_ATOM_TOKENS:
                        in_aromatic[i] = True

            ids  = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == EOS_IDX)
            if done.all():
                break

        # Post-process: fix any truncation edge cases
        results = []
        for i in range(n):
            tokens = ids[i].tolist()[1:]
            smi = ''
            for t in tokens:
                if t == EOS_IDX:
                    break
                if t not in (PAD_IDX, SOS_IDX) and t in idx2token:
                    smi += idx2token[t]

            # If we hit max_len without EOS, close open structure
            if EOS_IDX not in ids[i].tolist()[1:]:
                deficit = smi.count('(') - smi.count(')')
                if deficit > 0:
                    smi += ')' * deficit
                while smi and smi[-1] in ('=', '#', '-'):
                    smi = smi[:-1]

            results.append(smi)

        return results


# ── 3. Load weights ────────────────────────────────────────────────────────────
model = MolTransformer(vocab_size=VOCAB_SIZE).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
print("✅ MolTransformer loaded")


# ── 4. Strict validity using SanitizeMol ──────────────────────────────────────
def is_valid(smiles):
    """
    Two-stage check:
    - MolFromSmiles(sanitize=False): catches syntax errors (parse errors 3 & 4)
    - SanitizeMol: catches kekulization (error 1) and valence violations (error 2)
    """
    if not smiles or len(smiles) < 3:
        return False
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return False
    try:
        SanitizeMol(mol)
        return True
    except Exception:
        return False


# ── 5. Helpers ─────────────────────────────────────────────────────────────────
def smiles_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    img = Draw.MolToImage(mol, size=(400, 400))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def smiles_to_3d_sdf(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    try:
        mol    = AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result == -1:
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result == -1:
            return None
        AllChem.MMFFOptimizeMolecule(mol)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None

def lipinski_check(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"valid": False}
    mw   = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    qed  = round(QED.qed(mol), 3)
    return {
        "valid": True,
        "molecular_weight": mw,
        "logP": logp,
        "h_bond_donors": hbd,
        "h_bond_acceptors": hba,
        "qed": qed,
        "drug_like": (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10),
    }


# ── 6. /predict ────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data   = request.get_json()
    smiles = data.get("smiles", "").strip()

    if not smiles:
        return jsonify({"error": "SMILES not provided"}), 400
    if not is_valid(smiles):
        return jsonify({"error": "Invalid SMILES. Try: CC(=O)OC1=CC=CC=C1C(=O)O"}), 400

    try:
        input_image = smiles_to_image(smiles)
        input_props = lipinski_check(smiles)
        input_sdf   = smiles_to_3d_sdf(smiles)

        gen_smiles = None
        for attempt in range(10):
            temp       = 1.0 + (attempt * 0.1)
            candidates = model.generate(n=30, temperature=temp, top_k=10)
            valid      = [s for s in candidates if is_valid(s)]
            if valid:
                gen_smiles = valid[0]
                break

        if not gen_smiles:
            return jsonify({"error": "Model failed to generate a valid molecule. Try again."}), 500

        return jsonify({
            "input": {
                "smiles":     smiles,
                "image":      input_image,
                "properties": input_props,
                "sdf":        input_sdf,
            },
            "generated": {
                "smiles":     gen_smiles,
                "image":      smiles_to_image(gen_smiles),
                "properties": lipinski_check(gen_smiles),
                "sdf":        smiles_to_3d_sdf(gen_smiles),
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 7. /health ─────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "MolTransformer v2", "vocab": VOCAB_SIZE})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
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
