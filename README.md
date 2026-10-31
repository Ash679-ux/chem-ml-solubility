# Can you predict how well a molecule dissolves in water using machine learning?

I'm a second-year Chemistry with Maths student at UCL, and I wanted a small project that put machine learning to work on a real chemistry question. Solubility is a good one: it matters for drug design and environmental chemistry, and chemists already have decent rules of thumb for it (polar groups help, big greasy aromatic systems hurt). So the question is how much of that a model can pick up from structure alone.

The quantity I'm predicting is **logS**: the base-10 logarithm of a molecule's solubility in water, in moles per litre. So logS = -3 means 0.001 mol/L, and a higher (less negative) value means more soluble.

![Sample molecules](sample_molecules.png)

*A random sample of 16 molecules from the dataset, labelled with their measured logS. Small polar molecules sit near or above zero, while the least soluble in this sample (about -3.7 to -4.2) are a steroid, a chloro-iodobenzene and a small alkane.*

## Where I've got to

So far I have:

- loaded the ESOL (Delaney) dataset and turned every SMILES string (a line of text that encodes a molecule's structure, such as `CCO` for ethanol) into an RDKit molecule (RDKit is a chemistry toolkit for Python)
- checked the data: 1,128 molecules, no missing values, no duplicate SMILES, and every structure parses
- looked at the spread of the target (logS runs from about -11.6 to +1.6, averaging around -3)
- calculated seven RDKit descriptors (numbers worked out from a structure, such as molecular weight) and fitted a baseline, linear regression and random forest
- compared a random split with a scaffold split, to see how much a model's score depends on test molecules resembling the training ones
- repeated that comparison with cross-validation (repeating the train/test split several times and averaging), so the scores don't depend on a single split
- looked at which descriptors matter and where the model goes wrong (see Results)
- tested whether element counts and Morgan fingerprints (a way of encoding which substructures a molecule contains) fix the errors on phosphorus compounds, using cross-validation (they don't)

Still to do:

- look more closely at the phosphorus compounds, since better features didn't fix them
## Results

I compared three models on seven RDKit descriptors: molecular weight, logP (how much a molecule prefers fat-like surroundings to water-like ones), hydrogen bond donors and acceptors, rotatable bonds (a measure of flexibility), polar surface area (TPSA, the surface area contributed by polar atoms such as O and N) and the proportion of aromatic atoms.

- **Predict the mean** always guesses the average logS of the training set. It's the floor any real model has to beat.
- **Linear regression** fits a weighted sum of the descriptors.
- **Random forest** averages the predictions of many decision trees.

Each was tested two ways:

- **Random split:** 80% of the molecules chosen at random for training, and the other 20% held back for testing.
- **Scaffold split:** molecules that share a ring framework (the molecule with its side chains stripped off, called a Murcko scaffold) stay together on the same side. The test set therefore contains structures the model hasn't seen.

| Model | Split | RMSE | R² |
|---|---|---|---|
| Predict the mean | random | 2.174 | 0.000 |
| Linear regression | random | 1.093 | 0.747 |
| Random forest | random | 0.818 | 0.859 |
| Predict the mean | scaffold | 2.244 | -0.199 |
| Linear regression | scaffold | 1.289 | 0.605 |
| Random forest | scaffold | 0.955 | 0.783 |

**How to read the table.** RMSE (root mean squared error) is the typical size of a prediction error, in logS units, so lower is better. It penalises big misses more than small ones. An RMSE of 0.8 means predictions are typically about 0.8 log units off, which is roughly a factor of 6 in solubility. R² is the fraction of the variation in logS that the model explains: 1 is perfect, 0 is no better than always guessing the average, and a negative value is worse than that.

The random forest beat linear regression on both splits. Its RMSE rose from 0.82 on the random split to 0.96 on the scaffold split, so a random split overestimates how well the model handles new chemistry. The scaffold test set is not the same difficulty as the random one (the "predict the mean" baseline gets a negative R² on it), so RMSE is the fairer number to compare across the two splits.

**Checking with cross-validation.** Those numbers come from one split, so I repeated the comparison with 5-fold cross-validation (splitting the data into five parts and predicting each part with a model trained on the other four), and repeated that with three different shuffles of the folds. In the scaffold version, molecules that share a ring framework always stay in the same fold, and each ring-free molecule counts as its own group. The standard deviation across the shuffles was 0.007 or less for every number.

| Model | Split | RMSE | R² |
|---|---|---|---|
| Predict the mean | random | 2.098 | -0.003 |
| Linear regression | random | 0.995 | 0.774 |
| Random forest | random | 0.709 | 0.885 |
| Predict the mean | scaffold | 2.115 | -0.018 |
| Linear regression | scaffold | 1.023 | 0.762 |
| Random forest | scaffold | 0.804 | 0.853 |

The main conclusions hold: the random forest is clearly the best model, and its error rises on unseen ring frameworks (RMSE 0.709 to 0.804, about 13%). Two things change. Errors are lower across the board than on the single split (0.709 against 0.818 for the forest on a random split), so that split happened to be a harder-than-average test set. And under cross-validation linear regression barely gets worse on unseen frameworks (0.995 to 1.023, about 3%), whereas the single split suggested it suffered as much as the forest did. The scaffold cross-validation is also a milder test than the single scaffold split, which deliberately left the rarest frameworks for testing, so 0.804 and 0.955 aren't directly comparable.

**Which descriptors matter.** In the random forest, logP carries most of the importance (about 0.82), followed by molecular weight (about 0.10) and polar surface area (about 0.04). The other four are each below 0.02. (Importance is a score for how much the forest relies on each descriptor, and the scores add up to 1.) That fits the chemistry: solubility and lipophilicity are closely linked.

![Predicted vs measured logS](predicted_vs_measured.png)

*Random forest on the random-split test set. Points on the dashed line are perfect predictions.*

**Where it goes wrong.** The cluster of points well below the line on the right is four phosphorus compounds (measured logS 0.4 to 1.1, predicted -1.8 to -3.3), and they are four of the six largest errors. On the random-split test set, the 8 phosphorus-containing molecules had a mean absolute error (MAE, the average size of the error ignoring its direction) of 1.94 log units, compared with 0.56 across all 226 test molecules (roughly 0.5 for the other 218). The model tended to predict them as less soluble than measured. My first guess was that none of the seven descriptors encodes phosphorus directly, so I tested that next.

**Can richer features fix it?** I added element counts (how many C, N, O, S, P, F, Cl, Br and I atoms a molecule has, so the model can see phosphorus directly) and Morgan count fingerprints (1,024 bins recording which small substructures, such as P=O groups, a molecule contains). The test set held only 8 phosphorus compounds, so I compared the feature sets with 5-fold cross-validation (splitting the data into five parts and predicting each part with a model trained on the other four) over all 1,128 molecules. That gives every one of the 43 phosphorus compounds a prediction from a model that never saw it. I repeated it with three different shuffles of the folds. The values are means over the three repeats, and the shuffles moved them by about 0.03 or less, apart from the element-count bias (about 0.06).

| Features | Number of features | RMSE (all molecules) | MAE, no phosphorus | MAE, phosphorus (n=43) | Bias, phosphorus |
|---|---|---|---|---|---|
| 7 descriptors (baseline) | 7 | 0.709 | 0.488 | 1.000 | +0.610 |
| + element counts | 16 | 0.686 | 0.475 | 1.010 | +0.415 |
| + Morgan counts | 1,031 | 0.681 | 0.475 | 0.982 | +0.540 |
| + both | 1,040 | 0.675 | 0.472 | 0.953 | +0.503 |

*Bias is the average of measured minus predicted logS for the phosphorus compounds, so a positive value means they are predicted as less soluble than they really are.*

![Mean absolute error by feature set](feature_comparison.png)

They didn't fix it. Overall error fell slightly (RMSE 0.709 to 0.675 with both added), but the phosphorus error barely moved. It stays about 1.0 log units whatever features the model gets, roughly twice the error on everything else (about 0.47 to 0.49). Element counts made the phosphorus errors less one-sided (bias +0.61 down to +0.42) without making them any smaller. The best case, both feature sets together, lowered the phosphorus error from 1.00 to 0.95, which is too small a change to read much into with 43 molecules. So my first guess isn't supported: letting the model see phosphorus directly doesn't help. Two possible reasons, which I haven't tested, are that each training fold contains only about 34 phosphorus compounds for the forest to learn from, and that the measured values for some of these compounds may be noisy.

Cross-validation also puts the single-split numbers in context. Across all molecules the baseline forest's RMSE is 0.709, lower than the 0.818 on the single test split, and its phosphorus error is 1.00 rather than 1.94. The single test set happened to contain harder-than-average phosphorus compounds, so it overstated the problem, although phosphorus compounds are still the hardest group.

**Caveats.** The first model comparison table uses a single split and seed, so the cross-validated version is the more reliable one. Even in the cross-validation, having only 43 phosphorus compounds adds more uncertainty than the spread between fold shuffles shows, so small differences in the feature table aren't meaningful. I also chose the new features after seeing where the model failed.

## The data

ESOL is a well-known benchmark of 1,128 small molecules with measured aqueous solubility, given as SMILES strings and logS. The code downloads the DeepChem copy the first time you run it and saves it as `delaney-processed.csv`, so after that it works offline.

Delaney, J. S. *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure.* J. Chem. Inf. Comput. Sci. 2004, 44, 1000-1005.

## What's in the repo

- `Solubility.py` has the helper functions `load_data()` and `explore()`
- `molecules.ipynb` is the notebook: it loads and checks the data, computes descriptors, fits the models, compares the two splits, analyses the errors, runs the feature comparison and cross-validates the main comparison
- `delaney-processed.csv` is the cached dataset
- `results.csv` is the single-split model comparison
- `results_cv.csv` is the cross-validated model comparison, with the spread across repeats
- `results_features.csv` is the cross-validated feature comparison, with the spread across repeats
- `sample_molecules.png`, `predicted_vs_measured.png` and `feature_comparison.png` are the three figures
- `requirements.txt` lists the Python packages

## Running it yourself

You'll need Python 3.

```bash
git clone https://github.com/Ash679-ux/chem-ml-solubility.git
cd chem-ml-solubility

python3 -m venv .venv
source .venv/bin/activate      # on Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Then open `molecules.ipynb` in VS Code or Jupyter and run all the cells from the top. The results are reproducible, because every random step uses a fixed seed. You can also use the helper functions directly:

```python
from Solubility import load_data, explore

df = load_data()   # columns: smiles, logS, mol
explore(df)        # prints summary stats and saves sample_molecules.png
```

## Tools

Python, pandas, NumPy, scikit-learn, RDKit, Jupyter and Matplotlib.

## About me

I'm Ashwin, looking for chemistry-related or quantitative analysis internships for summer 2027. If you've got feedback on the project or want to talk about it, I'd be glad to hear from you.