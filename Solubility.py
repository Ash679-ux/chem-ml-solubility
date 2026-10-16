
#Install:  pip install rdkit pandas numpy

import os

import numpy as np
import pandas as pd
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
CACHE = "delaney-processed.csv"
SEED = 42


# load and explore
def load_data():
    if not os.path.exists(CACHE):
        pd.read_csv(URL).to_csv(CACHE, index=False)
    df = pd.read_csv(CACHE)
    df = df.rename(columns={"measured log solubility in mols per litre": "logS"})
    df = df[["smiles", "logS"]].copy()
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["mol"]).reset_index(drop=True)
    return df


def explore(df):
    print(df.shape)
    print(df["logS"].describe())
    s = df.sample(16, random_state=0)
    img = Draw.MolsToGridImage(s["mol"].tolist(), molsPerRow=4,
                           legends=[f"logS={x:.2f}" for x in s["logS"]])
    if hasattr(img, "save"):          # PIL image (plain Python)
        img.save("sample_molecules.png")
    else:                             # IPython Image (Jupyter): raw PNG bytes
        with open("sample_molecules.png", "wb") as f:
            f.write(img.data)