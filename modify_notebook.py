"""
modify_notebook.py — patches molgen.ipynb in-place.

Fixes applied:
  1. Moves the beta/alpha diffusion schedule tensors to `device`.
  2. Replaces the slow per-molecule nearest_smiles_from_fp with a
     vectorised BulkTanimotoSimilarity version.

Run once after generating / editing the notebook:
    python modify_notebook.py
"""

import json

NOTEBOOK = "molgen.ipynb"   # ← corrected filename (was "Drug_Discovery.ipynb")

with open(NOTEBOOK, "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue

    source = cell.get("source", [])
    source_str = "".join(source)

    # ── Fix 1: move betas to device ──────────────────────────────────────────
    new_source = []
    for line in source:
        if line.strip() == "betas = torch.linspace(0.0001, 0.02, timesteps)":
            line = line.replace(
                "betas = torch.linspace(0.0001, 0.02, timesteps)",
                "betas = torch.linspace(0.0001, 0.02, timesteps).to(device)",
            )
        new_source.append(line)
    cell["source"] = new_source
    source = new_source

    # ── Fix 2: replace slow nearest_smiles_from_fp ───────────────────────────
    if "def nearest_smiles_from_fp(" in "".join(source):
        cell["source"] = [
            "from rdkit import DataStructs\n",
            "import numpy as np\n",
            "\n",
            "def nearest_smiles_from_fp(fp_generated, fps_real, smiles_real):\n",
            "    \"\"\"\n",
            "    Vectorised nearest-neighbour search using BulkTanimotoSimilarity.\n",
            "    fp_generated : list/tensor of float arrays, each shape [2048]\n",
            "    fps_real     : list of uint8 numpy arrays, shape [2048] each\n",
            "    smiles_real  : list of SMILES strings, same length as fps_real\n",
            "    \"\"\"\n",
            "    # Pre-build RDKit ExplicitBitVect objects for the reference set\n",
            "    ref_fps = []\n",
            "    for real_fp in fps_real:\n",
            "        bin_str = ''.join('1' if b else '0' for b in real_fp)\n",
            "        ref_fps.append(DataStructs.CreateFromBitString(bin_str))\n",
            "\n",
            "    closest_smiles = []\n",
            "    for fp in fp_generated:\n",
            "        arr = fp.cpu().numpy() if hasattr(fp, 'cpu') else np.array(fp)\n",
            "        bin_str = ''.join('1' if b > 0.5 else '0' for b in arr)\n",
            "        try:\n",
            "            gen_rdkit = DataStructs.CreateFromBitString(bin_str)\n",
            "        except Exception as e:\n",
            "            print('⚠️ Fingerprint error:', e)\n",
            "            continue\n",
            "\n",
            "        # BulkTanimotoSimilarity returns a Python list of floats\n",
            "        sims     = DataStructs.BulkTanimotoSimilarity(gen_rdkit, ref_fps)\n",
            "        best_idx = int(np.argmax(sims))\n",
            "        closest_smiles.append(smiles_real[best_idx])\n",
            "\n",
            "    return closest_smiles\n",
        ]

with open(NOTEBOOK, "w") as f:
    json.dump(nb, f, indent=2)

print(f"✅ {NOTEBOOK} patched successfully.")