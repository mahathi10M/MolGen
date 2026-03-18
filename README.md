# 🧬 MolGen — AI Drug Molecule Generator

A web application that uses a **PyTorch diffusion model** to generate novel drug-like molecules from a SMILES input. Built with Flask, RDKit, and a React frontend.

---

## 🖥️ Demo

Enter any SMILES string → the diffusion model generates a new molecule → view it in 2D/3D and check Lipinski drug-likeness properties.

---

## 📁 Project Structure

```
molgen/
├── app.py                  # Flask backend (API server)
├── model_loader.py         # DiffusionModel architecture + loader
├── preprocessor.py         # SMILES → Morgan fingerprint (2048-bit)
├── predict.py              # Standalone prediction script
├── modify_notebook.py      # Patches molgen.ipynb in-place
├── molgen.ipynb            # Training notebook
├── model.pth               # Trained model weights
├── SMILES_Big_Data_Set.csv # Training dataset
└── index.html              # React frontend
```

---

## ⚙️ Requirements

- Python 3.8+
- pip packages:

```bash
pip install flask flask-cors torch rdkit pandas numpy
```

---

## 🚀 Running Locally

### 1. Start the Flask backend

```bash
python app.py
```

The server will start at `http://localhost:5001`

You should see:
```
📂 Loading training SMILES...
✅ Loaded 5000 training molecules
✅ Model loaded
```

### 2. Open the frontend

Just open `index.html` in your browser. No build step needed.

### 3. Test it

- Enter a SMILES string like `CC(=O)OC1=CC=CC=C1C(=O)O` (Aspirin)
- Or click one of the quick example pills
- Hit **⚗ Generate**

---

## 🔬 How It Works

```
Input SMILES
     ↓
Morgan Fingerprint (2048 bits)
     ↓
Forward Diffusion (add noise, t=800 steps)
     ↓
Reverse Diffusion (denoise, DiffusionModel)
     ↓
Generated Fingerprint
     ↓
Nearest-neighbour Tanimoto search (BulkTanimotoSimilarity)
     ↓
Output SMILES + 2D image + 3D viewer + Lipinski properties
```

---

## 🌐 API Endpoints

### `POST /predict`

**Request:**
```json
{ "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O" }
```

**Response:**
```json
{
  "input": {
    "smiles": "...",
    "image": "<base64 PNG>",
    "properties": {
      "molecular_weight": 180.16,
      "logP": 1.31,
      "h_bond_donors": 1,
      "h_bond_acceptors": 3,
      "drug_like": true
    },
    "sdf": "..."
  },
  "generated": {
    "smiles": "...",
    "image": "<base64 PNG>",
    "properties": { ... },
    "sdf": "..."
  }
}
```

### `GET /health`

```json
{ "status": "ok", "molecules": 5000 }
```

---

## 🧪 Standalone Prediction (no server)

```bash
python predict.py "CC(=O)OC1=CC=CC=C1C(=O)O"
```

---

## 🏋️ Training Results (Google Colab)

### Dataset
| Property | Value |
|----------|-------|
| Source | Kaggle — `yanmaksi/big-molecules-smiles-dataset` |
| Total SMILES loaded | 15,872 |
| Valid fingerprints | 15,872 × 2048 bits |
| Fingerprint speed | 610 molecules/sec |

### Training
| Property | Value |
|----------|-------|
| Device | CPU |
| Epochs | 100 |
| Total parameters | 4,925,312 |
| Starting loss | 0.797866 (epoch 10) |
| Final loss | 0.783668 (epoch 100) |
| **Best loss** | **0.782770 (epoch 91)** |
| LR schedule | Cosine decay 2e-4 → 0 |

**Loss curve:**
```
Epoch  10 │ 0.797866  ████████████████████
Epoch  20 │ 0.791337  ███████████████████
Epoch  30 │ 0.789043  ███████████████████
Epoch  40 │ 0.787415  ██████████████████
Epoch  50 │ 0.785937  ██████████████████
Epoch  60 │ 0.784966  ██████████████████
Epoch  70 │ 0.784701  ██████████████████
Epoch  80 │ 0.783935  ██████████████████
Epoch  90 │ 0.783432  ██████████████████
Epoch 100 │ 0.783668  ██████████████████
```

### Evaluation (200 sampled molecules)
| Metric | Score | Meaning |
|--------|-------|---------|
| **Validity** | 27.0% | Chemically valid SMILES |
| **Uniqueness** | 27.0% | Non-duplicate molecules |
| **Novelty** | 0.0% | Not seen in training set |
| **Avg QED** | 0.330 | Drug-likeness score (0–1) |
| Sampling speed | 2.43 mol/sec | On CPU |

> **Note on Novelty 0%:** The model currently retrieves the nearest training molecule via Tanimoto similarity rather than decoding the fingerprint into a brand-new SMILES. A graph neural network decoder would be needed to generate truly novel structures.

---

## 📊 Model Details

| Property | Value |
|----------|-------|
| Architecture | Diffusion Model |
| Input dim | 2048 (Morgan fingerprint) |
| Hidden dim | 1024 |
| Time embedding | Sinusoidal (128-dim) |
| Timesteps | 1000 |
| Training data | ~16K drug-like SMILES |
| Framework | PyTorch |

---

## 💊 Lipinski Rule of Five (Ro5)

The app checks if generated molecules are **drug-like** using Lipinski's rules:

| Property | Limit |
|----------|-------|
| Molecular Weight | ≤ 500 Da |
| logP | ≤ 5 |
| H-Bond Donors | ≤ 5 |
| H-Bond Acceptors | ≤ 10 |

---

## 🚢 Deployment

### Railway (recommended)

1. Push your project to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Set start command: `python app.py`
4. Update `BACKEND` in `index.html` from `http://localhost:5001` to your Railway URL
5. Deploy `index.html` separately on [Netlify](https://netlify.com) (drag & drop)

### Render

1. Go to [render.com](https://render.com) → New Web Service
2. Connect your GitHub repo
3. Build command: `pip install flask flask-cors torch rdkit pandas numpy`
4. Start command: `python app.py`
5. Update the backend URL in `index.html`

---

## 🔍 Limitations & Next Steps

The model achieves 27% chemical validity and 0.330 QED on CPU-only training over 100 epochs. Novelty is 0% because the pipeline retrieves the nearest training molecule via Tanimoto similarity — a known limitation of fingerprint-based diffusion without a SMILES decoder.

**What I'd improve next:**
- Add a GNN or VAE decoder to generate truly novel SMILES instead of retrieving nearest neighbours
- Train on GPU for more epochs to push validity above 80%
- Use reinforcement learning (REINFORCE) to directly optimise QED and drug-likeness
- Switch to a graph-based diffusion model (e.g. GDSS) for better chemical validity

---

## ⚠️ Known Limitations

- Generation is slow on CPU (~30s per molecule) — use GPU for faster inference
- The model retrieves the nearest training molecule rather than generating truly novel SMILES — a decoder (e.g. graph neural net) would be needed for fully novel structures
- `index.html` has the backend URL hardcoded — update it before deploying

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot connect to Flask backend` | Make sure `python app.py` is running on port 5001 |
| `Invalid SMILES` error | Check your SMILES string at [pubchem.ncbi.nlm.nih.gov](https://pubchem.ncbi.nlm.nih.gov) |
| `model.pth not found` | Make sure `model.pth` is in the same folder as `app.py` |
| `SMILES_Big_Data_Set.csv not found` | Place the CSV in the same folder as `app.py` |
| 3D viewer not loading | It loads from a CDN — check your internet connection |

---

