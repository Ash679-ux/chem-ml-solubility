# Can you predict how well a molecule dissolves in water using machine learning?

I'm a second-year Chemistry with Maths student at UCL, and I wanted a small project that put machine learning to work on a real chemistry question. Solubility is a good one: it matters for drug design and environmental chemistry, and chemists already have decent rules of thumb for it (polar groups help, big greasy aromatic systems hurt). So the question is how much of that a model can pick up from structure alone.

![Sample molecules](sample_molecules.png)

*A random sample of 16 molecules from the dataset, labelled with their measured logS. The flat, ring-heavy hydrocarbons sit at the insoluble end, as expected.*

## Where I've got to

This is a work in progress. So far I have:

- loaded the ESOL (Delaney) dataset and turned every SMILES string into an RDKit molecule
- checked the data: 1,128 molecules, no missing values, no duplicates, and every structure parses
- looked at the spread of the target (logS runs from about -11.6 to +1.6, averaging around -3)
- drawn a sample of structures

Still to do:

- calculate molecular descriptors (molecular weight, logP, hydrogen bond donors and acceptors, and so on)
- fit a baseline model, probably a random forest
- compare a random train/test split with a scaffold split, to see how much a model's score depends on test molecules resembling the training ones
- write up what I find

## The data

ESOL is a well-known benchmark of 1,128 small molecules with measured aqueous solubility, given as SMILES strings and logS (the log of solubility in mol/L). The code downloads the DeepChem copy the first time you run it and saves it as `delaney-processed.csv`, so after that it works offline.

Delaney, J. S. *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure.* J. Chem. Inf. Comput. Sci. 2004, 44, 1000-1005.

## What's in the repo

- `Solubility.py` has two helper functions: `load_data()` and `explore()`
- `molecules.ipynb` is the notebook where I load, check and look at the data
- `delaney-processed.csv` is the cached dataset
- `sample_molecules.png` is the picture above, made by `explore()`
- `requirements.txt` lists the Python packages

## Running it yourself

Python 3 is required

```bash
git clone https://github.com/Ash679-ux/chem-ml-solubility.git
cd chem-ml-solubility

python3 -m venv .venv
source .venv/bin/activate      # on Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Then open `molecules.ipynb` and run the cells, or use the functions directly:

```python
from Solubility import load_data, explore

df = load_data()   # columns: smiles, logS, mol
explore(df)        # prints summary stats and saves sample_molecules.png
```

## Tools

Python, pandas, NumPy, scikit-learn, RDKit, Jupyter and Matplotlib.

## About me

I'm Ashwin, looking for chemistry-related or quantitative analysis internships for summer 2027. If you've got feedback on the project or want to talk about it, I'd be glad to hear from you.