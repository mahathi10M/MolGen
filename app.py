import warnings, os, math, json

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem, QED
from rdkit.Chem.rdmolops import AddHs
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app, origins="*")

device = torch.device("cpu")

# ── 1. Load checkpoint (weights + tokenizer bundled together) ─────────────────
ckpt = torch.load("model_v2.pth", map_location=device, weights_only=False)

token2idx  = ckpt['token2idx']
idx2token  = {int(i): t for t, i in token2idx.items()}
VOCAB_SIZE = ckpt['vocab_size']
PAD_IDX    = ckpt['pad_idx']
SOS_IDX    = ckpt['sos_idx']
EOS_IDX    = ckpt['eos_idx']
MAX_LEN    = 120

print(f"Tokenizer loaded | vocab={VOCAB_SIZE} PAD={PAD_IDX} SOS={SOS_IDX} EOS={EOS_IDX}")

# ── 2. Model — must exactly match Cell 6 of the training notebook ────────────
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
        B, T = x.shape
        emb   = self.pos_enc(self.embedding(x) * math.sqrt(self.d_model))
        mask  = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        pad_m = (x == PAD_IDX)
        mem   = self._mem_dummy.expand(B, -1, -1)
        out   = self.transformer(emb, mem, tgt_mask=mask, tgt_key_padding_mask=pad_m)
        return self.output(out)

    @torch.no_grad()
    def generate(self, n=10, temperature=1.0, top_k=10):
        self.eval()
        UNK_IDX = token2idx.get('<UNK>', -1)
        ids  = torch.full((n, 1), SOS_IDX, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)

        for _ in range(MAX_LEN):
            logits = self.forward(ids)[:, -1, :] / max(temperature, 0.1)
            logits[:, PAD_IDX] = float('-inf')
            logits[:, SOS_IDX] = float('-inf')
            if UNK_IDX >= 0:
                logits[:, UNK_IDX] = float('-inf')
            if top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits  = logits.masked_fill(logits < vals[:, -1:], float('-inf'))
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=1.0 / probs.size(-1))
            probs = probs / probs.sum(dim=-1, keepdim=True)
            nxt   = torch.multinomial(probs, 1).squeeze(1)
            nxt   = nxt.masked_fill(done, PAD_IDX)
            ids   = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            done  = done | (nxt == EOS_IDX)
            if done.all():
                break

        results = []
        for i in range(n):
            chars = []
            for t in ids[i].tolist()[1:]:
                if t == EOS_IDX:
                    break
                if t != PAD_IDX and t in idx2token:
                    chars.append(idx2token[t])
            results.append(''.join(chars))
        return results


# ── 3. Load weights ────────────────────────────────────────────────────────────
model = MolTransformer(vocab_size=VOCAB_SIZE).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
print("MolTransformer loaded and ready")

# ── 4. Helpers ──────────────────────────────────────────────────────────────────
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

MAX_INPUT_LEN = 150  # guards against pathological strings hanging RDKit's embedder

# ── 5. /predict ───────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data   = request.get_json(silent=True) or {}
    smiles = str(data.get("smiles", "")).strip()

    if not smiles:
        return jsonify({"error": "SMILES not provided"}), 400
    if len(smiles) > MAX_INPUT_LEN:
        return jsonify({"error": f"SMILES too long (max {MAX_INPUT_LEN} characters)"}), 400
    if not Chem.MolFromSmiles(smiles):
        return jsonify({"error": "Invalid SMILES. Try: CC(=O)OC1=CC=CC=C1C(=O)O"}), 400

    try:
        input_image = smiles_to_image(smiles)
        input_props = lipinski_check(smiles)
        input_sdf   = smiles_to_3d_sdf(smiles)

        gen_smiles = None
        for attempt in range(10):
            temp = 1.0 + (attempt * 0.1)   # 1.0, 1.1, 1.2 ... 1.9
            candidates = model.generate(n=30, temperature=temp, top_k=10)
            valid = [s for s in candidates if Chem.MolFromSmiles(s)]
            if valid:
                gen_smiles = valid[0]
                break

        if not gen_smiles:
            return jsonify({"error": "Model failed to generate a valid molecule. Try again."}), 500

        return jsonify({
            "input": {
                "smiles": smiles,
                "image":  input_image,
                "properties": input_props,
                "sdf":    input_sdf,
            },
            "generated": {
                "smiles": gen_smiles,
                "image":  smiles_to_image(gen_smiles),
                "properties": lipinski_check(gen_smiles),
                "sdf":    smiles_to_3d_sdf(gen_smiles),
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── 6. /health ────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "MolTransformer v2", "vocab": VOCAB_SIZE})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
