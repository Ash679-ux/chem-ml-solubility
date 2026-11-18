"""Helpers for the aqueous solubility project.

Everything the notebook reuses lives here: data loading, the seven
descriptors, fingerprints, phosphorus classes and the two scaffold splits.

Install the dependencies with:  pip install -r requirements.txt
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

HERE = Path(__file__).resolve().parent

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
CACHE = HERE / "delaney-processed.csv"
AQSOLDB_URL = "https://raw.githubusercontent.com/mcsorkun/AqSolDB/master/results/data_curated.csv"
AQSOLDB_CSV = HERE / "aqsoldb.csv"
ULRICH_URL = "https://raw.githubusercontent.com/nadinulrich/log_Sw_prediction/main/dataset.xlsx"
ULRICH_CSV = HERE / "ulrich.csv"

SEED = 42  # used for model training and train/test splits
CV_SEEDS = [SEED, SEED + 1, SEED + 2]  # three repeats of cross-validation
SAMPLE_SEED = 0  # fixed so sample_molecules.png shows the same molecules every run


# ---------------------------------------------------------------- ESOL

def load_data():
    """Return ESOL as a DataFrame with columns smiles, logS and mol (an RDKit molecule).

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


# ---------------------------------------------------------------- features

def featurize(mol):
    """Seven descriptors a chemist would expect to matter for solubility."""
    heavy = mol.GetNumHeavyAtoms()
    aromatic = sum(a.GetIsAromatic() for a in mol.GetAtoms())
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "AromProp": aromatic / heavy if heavy else 0.0,  # fraction of heavy atoms that are aromatic
    }


# radius 2: each bin counts an atom together with its neighbours up to two bonds away
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def morgan_counts(mols):
    """Morgan count fingerprints (radius 2, 1,024 bins), one row per molecule."""
    return pd.DataFrame(
        np.stack([MORGAN.GetCountFingerprintAsNumPy(m) for m in mols]),
        columns=[f"fp{i}" for i in range(1024)],
    )


# ---------------------------------------------------------------- phosphorus

# P=O and P=S groups, however the SMILES draws them: as a double bond,
# or as P+ single-bonded to O- / S- (some entries in the larger datasets use this form).
PO_PATTERNS = [Chem.MolFromSmarts("[#15]=[#8]"), Chem.MolFromSmarts("[#15+]-[#8-]")]
PS_PATTERNS = [Chem.MolFromSmarts("[#15]=[#16]"), Chem.MolFromSmarts("[#15+]-[#16-]")]


def count_matches(mol, patterns):
    """Total number of matches of any of the SMARTS patterns in mol."""
    return sum(len(mol.GetSubstructMatches(p)) for p in patterns)


def p_class(mol):
    """'P=O', 'P=S', 'other P' or 'no phosphorus'. P=O is checked first."""
    if count_matches(mol, PO_PATTERNS):
        return "P=O"
    if count_matches(mol, PS_PATTERNS):
        return "P=S"
    if any(a.GetSymbol() == "P" for a in mol.GetAtoms()):
        return "other P"
    return "no phosphorus"


# ---------------------------------------------------------------- splits

def murcko(mol):
    """Murcko scaffold SMILES: the ring systems and the chains linking them ('' if no rings)."""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_split(mols, frac_train=0.8):
    """Single train/test split with no Murcko scaffold on both sides.

    Groups are filled into training largest first; groups that would overflow
    it go to test. All ring-free molecules share the empty scaffold, which is
    usually the largest group, so they all end up in training.
    Returns (train_idx, test_idx).
    """
    groups = defaultdict(list)
    for i, m in enumerate(mols):
        groups[murcko(m)].append(i)
    # largest groups first, ties broken by index so the split is reproducible
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n_train = int(frac_train * len(mols))
    train_idx, test_idx = [], []
    for g in ordered:
        if len(train_idx) + len(g) <= n_train:
            train_idx.extend(g)
        else:
            test_idx.extend(g)
    return train_idx, test_idx


def scaffold_folds(mols, n_splits=5, seed=SEED):
    """Cross-validation folds where molecules sharing a ring framework stay in the same fold.

    Ring-free molecules share no framework, so each is its own group and they
    are spread across all folds. Returns a list of (train_idx, test_idx).
    """
    groups = defaultdict(list)
    for i, m in enumerate(mols):
        groups[murcko(m) or f"ring-free {i}"].append(i)
    keys = list(groups)
    fold_of = np.empty(len(mols), dtype=int)
    sizes = np.zeros(n_splits, dtype=int)
    for k in np.random.default_rng(seed).permutation(len(keys)):
        f = int(np.argmin(sizes))  # put each group in the currently smallest fold
        fold_of[groups[keys[k]]] = f
        sizes[f] += len(groups[keys[k]])
    idx = np.arange(len(mols))
    return [(idx[fold_of != f], idx[fold_of == f]) for f in range(n_splits)]


# ---------------------------------------------------------------- larger datasets

def prepare(frame):
    """Parse SMILES and add mol, InChIKey and phosphorus class.

    Rows RDKit can't parse are dropped. When the same molecule (same InChIKey)
    appears more than once, its logS values are averaged.
    """
    frame = frame[["smiles", "logS"]].copy()
    frame["mol"] = frame["smiles"].apply(Chem.MolFromSmiles)
    frame = frame.dropna(subset=["mol"])
    frame["inchikey"] = frame["mol"].apply(Chem.MolToInchiKey)
    frame = frame[frame["inchikey"] != ""]
    frame = (
        frame.groupby("inchikey", sort=False)
        .agg(smiles=("smiles", "first"), logS=("logS", "mean"), mol=("mol", "first"))
        .reset_index()
    )
    frame["p_class"] = frame["mol"].apply(p_class)
    return frame


def load_aqsoldb():
    """AqSolDB (Sorkun et al. 2019) without salts and mixtures, prepared as above."""
    if not AQSOLDB_CSV.exists():
        pd.read_csv(AQSOLDB_URL).to_csv(AQSOLDB_CSV, index=False)
    raw = pd.read_csv(AQSOLDB_CSV).rename(columns={"SMILES": "smiles", "Solubility": "logS"})
    raw = raw[~raw["smiles"].str.contains(".", regex=False)]  # drop salts and mixtures
    return prepare(raw)


def load_ulrich():
    """The curated set of Ulrich et al. (2025), prepared as above. Needs openpyxl."""
    if not ULRICH_CSV.exists():
        # the authors' spreadsheet; "log Sw dataset used" is their curated logS value
        (pd.read_excel(ULRICH_URL, sheet_name="dataset_split")
           .rename(columns={"SMILES": "smiles", "log Sw dataset used": "logS"})
           [["smiles", "logS"]]
           .to_csv(ULRICH_CSV, index=False))
    return prepare(pd.read_csv(ULRICH_CSV))
