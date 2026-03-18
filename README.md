## Drug Molecule Generator with Diffusion Model

This project presents a web-based platform that integrates a diffusion model trained on SMILES data to generate novel drug-like molecules. It features an interactive React frontend with 3D molecular visualization, allowing users to input molecules, visualize generated structures, and explore drug-likeness properties seamlessly.

## 🚀 Features

- **SMILES-Based Diffusion Model:** Generates novel molecular structures by learning from a dataset of SMILES representations.
- **Molecular Visualization:** Displays 2D structure images of both input and generated molecules.
- **Interactive 3D Viewer:** Spin and rotate molecules in 3D directly in the browser using 3Dmol.js.
- **Drug-Likeness Analysis:** Reports Lipinski's Rule of Five properties (MW, logP, H-bond donors/acceptors).
- **React Frontend:** A polished single-file HTML/React interface — no build step or npm required.

## 🧰 Technologies Used

- **Diffusion Models:** Implemented using PyTorch to learn and generate molecular structures from SMILES data.
- **Cheminformatics:** RDKit for SMILES parsing, fingerprint generation, and molecule rendering.
- **Backend:** Flask to handle API requests and model inference.
- **Frontend:** React (via CDN) + HTML/CSS — open `index.html` directly in any browser.
- **3D Visualization:** 3Dmol.js for interactive in-browser molecular rendering.

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip
- Any modern browser (Chrome, Firefox, Safari, Edge)

### Steps

```sh
# Step 1: Clone the repository
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory
cd <YOUR_PROJECT_NAME>

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the Flask backend
python app.py
```

The Flask backend runs at `http://localhost:5001`.

```sh
# Step 5: Open the frontend
open index.html   # macOS
# or just double-click index.html in Finder / Explorer
```

## 📦 Requirements

Add the following to your `requirements.txt`:

```
flask
flask-cors
torch
rdkit
numpy
pandas
```

## 🖥️ Usage

1. Make sure `python app.py` is running in a terminal (port 5001).
2. Open `index.html` in your browser.
3. Enter a valid SMILES string (e.g., `CC(=O)OC1=CC=CC=C1C(=O)O` for Aspirin) or click a quick example.
4. Click **Generate** to run the diffusion model.
5. View the input and generated molecule structures side by side.
6. Interact with the 3D viewer — drag to rotate, scroll to zoom.
7. Check the Lipinski drug-likeness properties for both molecules.
8. Download the generated molecule as an SDF file for use in PyMOL or UCSF Chimera.

## ✏️ Edit Files

**Directly in GitHub:**
- Navigate to the desired file.
- Click the "Edit" button (pencil icon) at the top right.
- Make your changes and commit.

**Using GitHub Codespaces:**
- Click the "Code" button on the repo page.
- Select the "Codespaces" tab and open a new Codespace.
- Edit files and commit directly from the browser.

## 🛠️ What technologies are used for this project?

- Python
- PyTorch
- RDKit
- Flask
- React (CDN)
- 3Dmol.js
- HTML / CSS / JavaScript
