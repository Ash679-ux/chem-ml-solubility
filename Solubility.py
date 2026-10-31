"""Helpers for the ESOL (Delaney) aqueous solubility project.

Install the dependencies with:  pip install -r requirements.txt
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
CACHE = Path(__file__).resolve().parent / "delaney-processed.csv"
SEED = 42  # used for model training and train/test splits
SAMPLE_SEED = 0  # fixed so sample_molecules.png shows the same molecules every run


def load_data():
    """Return a DataFrame with columns smiles, logS and mol (an RDKit molecule).

    The dataset is downloaded once and cached next to this file as
    delaney-processed.csv. Rows whose SMILES RDKit cannot parse are dropped.
    """
    if not CACHE.exists():
        pd.read_csv(URL).to_csv(CACHE, index=False)
    df = pd.read_csv(CACHE)
    df = df.rename(columns={"measured log solubility in mols per litre": "logS"})
    df = df[["smiles", "logS"]].copy()
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["mol"]).reset_index(drop=True)
    return df


def explore(df):
    """Print the shape and logS summary, and save sample_molecules.png.

    The picture is a 4x4 grid of randomly chosen molecules, each labelled
    with its measured logS. It is written to the current working directory.
    """
    print(df.shape)
    print(df["logS"].describe())
    s = df.sample(16, random_state=SAMPLE_SEED)
    img = Draw.MolsToGridImage(
        s["mol"].tolist(),
        molsPerRow=4,
        legends=[f"logS={x:.2f}" for x in s["logS"]],
    )
    if hasattr(img, "save"):  # PIL image (plain Python)
        img.save("sample_molecules.png")
    else:  # IPython Image (Jupyter): raw PNG bytes
        with open("sample_molecules.png", "wb") as f:
            f.write(img.data)