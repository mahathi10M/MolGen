"""
predict.py — standalone prediction using MolTransformer v2.
Usage:
    python predict.py "CC(=O)OC1=CC=CC=C1C(=O)O"
"""
import sys, json, math, re, torch
import torch.nn as nn
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import SanitizeMol

# Silence all RDKit stderr output
RDLogger.DisableLog('rdApp.*')

device = torch.device("cpu")

# ── Load checkpoint (tokenizer + weights bundled) ─────────────────────────────
ckpt = torch.load("model_v2.pth", map_location=device, weights_only=False)

token2idx = ckpt['token2idx']
idx2token = {int(i): t for t, i in token2idx.items()}
VOCAB_SIZE = ckpt['vocab_size']
PAD_IDX    = ckpt['pad_idx']
SOS_IDX    = ckpt['sos_idx']
EOS_IDX    = ckpt['eos_idx']
MAX_LEN    = 85

UNK_IDX   = token2idx.get('<UNK>', -1)
OPEN_IDX  = token2idx.get('(', -1)
CLOSE_IDX = token2idx.get(')', -1)

BOND_TOKENS = {'#', '=', '-'}
BOND_IDXS   = frozenset(token2idx[t] for t in BOND_TOKENS if t in token2idx)
DIGIT_IDXS  = {token2idx[d]: d for d in '1234567' if d in token2idx}

AROMATIC_ATOM_TOKENS = {'c', 'n', 'o', 's', '[nH]', '[n+]', '[o+]', '[se]'}
AROMATIC_IDXS = frozenset(token2idx[t] for t in AROMATIC_ATOM_TOKENS if t in token2idx)


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
        pad_m = (x == PAD_IDX).bool()
        mem   = self._mem_dummy.expand(B, -1, -1)
        out   = self.transformer(emb, mem, tgt_mask=mask, tgt_key_padding_mask=pad_m)
        return self.output(out)

    @torch.no_grad()
    def generate(self, n=5, temperature=1.0, top_k=10):
        self.eval()
        ids  = torch.full((n, 1), SOS_IDX, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)

        open_parens  = [0] * n
        open_rings   = [set() for _ in range(n)]
        pending_bond = [False] * n
        in_aromatic  = [False] * n

        for step in range(MAX_LEN):
            logits = self.forward(ids)[:, -1, :] / max(temperature, 0.1)
            logits[:, PAD_IDX] = float('-inf')
            logits[:, SOS_IDX] = float('-inf')
            if UNK_IDX >= 0:
                logits[:, UNK_IDX] = float('-inf')

            tokens_left = MAX_LEN - step

            for i in range(n):
                if done[i]:
                    continue

                if open_parens[i] == 0 and CLOSE_IDX >= 0:
                    logits[i, CLOSE_IDX] = float('-inf')
                if open_parens[i] > 0 or len(open_rings[i]) > 0:
                    logits[i, EOS_IDX] = float('-inf')
                if pending_bond[i]:
                    for bidx in BOND_IDXS:
                        logits[i, bidx] = float('-inf')
                    if CLOSE_IDX >= 0:
                        logits[i, CLOSE_IDX] = float('-inf')
                    logits[i, EOS_IDX] = float('-inf')
                    if OPEN_IDX >= 0:
                        logits[i, OPEN_IDX] = float('-inf')
                if not in_aromatic[i]:
                    for aidx in AROMATIC_IDXS:
                        logits[i, aidx] = float('-inf')
                for didx, dtok in DIGIT_IDXS.items():
                    if len(open_rings[i]) >= 4 and dtok not in open_rings[i]:
                        logits[i, didx] = float('-inf')
                if tokens_left <= 8:
                    if OPEN_IDX >= 0:
                        logits[i, OPEN_IDX] = float('-inf')
                    for bidx in BOND_IDXS:
                        logits[i, bidx] = float('-inf')
                    for didx, dtok in DIGIT_IDXS.items():
                        if dtok not in open_rings[i]:
                            logits[i, didx] = float('-inf')
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
                    open_parens[i] += 1; pending_bond[i] = False
                elif tok == ')':
                    open_parens[i] = max(0, open_parens[i] - 1); pending_bond[i] = False
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

        results = []
        for i in range(n):
            smi = ''
            for t in ids[i].tolist()[1:]:
                if t == EOS_IDX:
                    break
                if t not in (PAD_IDX, SOS_IDX) and t in idx2token:
                    smi += idx2token[t]
            if EOS_IDX not in ids[i].tolist()[1:]:
                deficit = smi.count('(') - smi.count(')')
                if deficit > 0:
                    smi += ')' * deficit
                while smi and smi[-1] in ('=', '#', '-'):
                    smi = smi[:-1]
            results.append(smi)
        return results


def is_valid(smiles):
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


# ── Load weights ──────────────────────────────────────────────────────────────
model = MolTransformer(vocab_size=VOCAB_SIZE).to(device)
model.load_state_dict(ckpt['model'])
model.eval()

if __name__ == "__main__":
    input_smiles = sys.argv[1] if len(sys.argv) > 1 else "CC(=O)OC1=CC=CC=C1C(=O)O"
    if not is_valid(input_smiles):
        print(f"❌ Invalid input SMILES: {input_smiles}")
        sys.exit(1)

    print(f"Input  : {input_smiles}")
    candidates = model.generate(n=20, temperature=1.0, top_k=10)
    valid = [s for s in candidates if is_valid(s)]
    print(f"Generated {len(valid)}/{len(candidates)} valid molecules:")
    for s in valid:
        print(f"  {s}")
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
