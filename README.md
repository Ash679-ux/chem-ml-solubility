# Can you predict how well a molecule dissolves in water using machine learning?

**In short:** a standard machine-learning model for solubility gets one group of compounds badly wrong: organophosphates with a P=O group, many of them pesticides. I tested why on two larger datasets: with enough examples the errors shrink, and adding an explicit count of P=O groups shrinks them further and removes most of the systematic bias, for compounds with and without rings.

![Error on the P=O compounds](po_external.png)

*Error on the 10 P=O compounds in the ESOL dataset. Grey: the original model and Delaney's published equation. Blue and orange: models trained on two larger datasets, without and with a P=O count.*

I'm a second-year Chemistry with Maths student at UCL. I wanted a project that put machine learning to work on a real chemistry question, and solubility is a good one: it matters for drug design and environmental chemistry, and chemists already have rules of thumb for it. The question is how much of that a model can learn from structure alone, and where it falls short.

The target is **logS**, the base-10 log of solubility in water in mol/L. So logS = -3 means 0.001 mol/L, and higher means more soluble.

**See it in action:** the follow-up app, [Molecule Property Explorer](https://github.com/Ash679-ux/Molecule-Property-Explorer), turns this into a web tool ([live demo](https://molecule-explorer.streamlit.app)).

## Key findings

- **A random forest on seven RDKit descriptors predicts logS with an RMSE of 0.71** in cross-validation (0.80 on ring frameworks it hasn't seen), against about 1.0 for linear regression and 2.1 for guessing the average.
- **Its worst failures are 10 phosphates and phosphonates** (P=O compounds), predicted 2.6 log units too insoluble on average. Element counts and fingerprints don't fix this, and Delaney's original equation gets the same compounds wrong.
- **More data plus a P=O count fixes most of it.** Trained on a larger dataset with every ESOL compound removed, the error on ESOL's P=O compounds falls from 1.85 to 0.79, better than Delaney's equation (1.40).
- **The count does more than generic fingerprints.** Across all 155 P=O compounds in a second dataset, it cuts their error by about 40% and the systematic bias by about two-thirds; fingerprints help less and leave nearly twice the bias.

## How I tested it

RDKit turns each molecule's SMILES string into seven descriptors: molecular weight, logP, hydrogen bond donors and acceptors, rotatable bonds, polar surface area and the proportion of aromatic atoms. Every model is scored with **5-fold cross-validation** repeated three times, so every molecule is predicted by a model that never saw it. **Scaffold folds** keep molecules sharing a ring framework together, testing the model on frameworks it hasn't seen.

**RMSE** and **MAE** are typical error sizes in logS units (lower is better). **Bias** is the average of measured minus predicted: positive means the model predicts compounds as less soluble than they are.

## Results

### Baselines

| Model | RMSE, random folds | RMSE, scaffold folds |
|---|---|---|
| Predict the mean | 2.098 | 2.115 |
| Linear regression | 0.995 | 1.023 |
| Random forest | **0.709** | **0.804** |

logP carries most of the forest's importance, which fits the chemistry.

![Predicted vs measured logS](predicted_vs_measured.png)

*Out-of-fold predictions for every molecule. The P=O compounds, in orange, sit well below the line: measured as soluble, predicted as much less soluble.*

### Where it fails

| Group | n | MAE: forest | MAE: Delaney's equation |
|---|---|---|---|
| No phosphorus | 1,085 | 0.49 | 0.70 |
| P=S (thiophosphates) | 33 | 0.55 | 0.47 |
| P=O (phosphates, phosphonates) | 10 | **2.62** | 1.40 |

Adding element counts and Morgan fingerprints barely helps: the phosphorus error stays around 1 log unit ([details](results_features.csv)).

**Why might P=O compounds be hard?** The P=O oxygen is an unusually strong hydrogen-bond acceptor, so water solvates these molecules far better than their size suggests, and the descriptors (including Crippen's logP) don't capture that. Several are also liquids or low-melting solids, which dissolve more readily than a model trained mostly on solids expects. And with only 10 examples, the model has little to learn from.

### More data, and a P=O count

I trained on two larger datasets, **AqSolDB** (about 10,000 compounds) and the curated set of **Ulrich et al.** (9,800), after removing every ESOL compound from them, including stereoisomers. That leaves 155 to 170 P=O compounds to learn from. I compared the seven descriptors with and without **nPO**, the number of P=O groups, and with and without Morgan fingerprints.

**Tested on ESOL's 10 P=O compounds** (trained on AqSolDB): 1.85 with the 7 descriptors, 1.37 with fingerprints, **0.79 with the P=O count**, 0.90 with both. Trained on Ulrich, the count gives 1.61: Ulrich et al. removed every compound with logS above 0.5, so it has few soluble P=O compounds to learn from.

**Across all 155 P=O compounds in the Ulrich set**, cross-validated:

| Features | P=O error, scaffold folds | Bias | Ring-free P=O compounds |
|---|---|---|---|
| 7 descriptors | 1.40 (1.26–1.55) | +1.26 | 1.54 |
| + P=O count | 0.86 (0.74–0.98) | +0.41 | 0.83 |
| + Morgan fingerprints | 0.98 (0.87–1.09) | +0.73 | 1.02 |
| + both | **0.80** (0.70–0.90) | **+0.37** | **0.80** |

*Brackets are 95% bootstrap intervals over the P=O compounds. Random folds give almost the same numbers.*

More data helps on its own, but the count helps more: lower error, about half the bias of fingerprints, and it works for ring-free compounds, which make up 8 of ESOL's 10 problem compounds. Error on all other molecules is unchanged by the count. Some under-prediction remains (bias about +0.4), which fits the liquids explanation above.

## Limitations

- **Small numbers.** ESOL's P=O result rests on 10 compounds, which were also used to find the problem. The Ulrich cross-validation is the stronger evidence.
- **Count vs fingerprints.** The bias difference is clear (non-overlapping intervals), but the error difference is smaller than the intervals, so it's suggestive rather than proven.
- **Imperfect overlap removal.** Matching by InChIKey connectivity catches stereoisomers but not salt forms or tautomers, and different datasets can disagree on the same compound.
- **Overall accuracy wasn't the focus.** I kept to simple features and a random forest so the results stay interpretable. Stronger models, such as gradient boosting or graph neural networks, would likely lower RMSE further.

## Data

- **ESOL:** Delaney, J. S. *J. Chem. Inf. Comput. Sci.* 2004, 44, 1000-1005 (DeepChem copy, cached as `delaney-processed.csv`)
- **AqSolDB:** Sorkun, M. C.; Khetan, A.; Er, S. *Sci. Data* 2019, 6, 143
- **Ulrich et al.:** Ulrich, N.; Voigt, K.; Kudria, A.; Böhme, A.; Ebert, R.-U. *J. Cheminform.* 2025, 17, 55

AqSolDB and the Ulrich set are downloaded from their authors' GitHub repositories the first time the notebook runs.

## What's in the repo

- `molecules.ipynb`: the whole analysis, in seven sections
- `Solubility.py`: reusable functions (data loading, descriptors, fingerprints, phosphorus classes, scaffold folds)
- `results_cv.csv`, `results_features.csv`, `results_external.csv`, `results_cv_ulrich.csv`: results for sections 3, 5, 6 and 7
- four figures, `requirements.txt` (pinned versions) and `LICENSE`

## Running it yourself

```bash
git clone https://github.com/Ash679-ux/chem-ml-solubility.git
cd chem-ml-solubility
python3 -m venv .venv
source .venv/bin/activate      # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Open `molecules.ipynb` and run all cells. Every random step uses a fixed seed. The first run takes from about 30 minutes to over an hour, and needs an internet connection to download the two larger datasets. It saves the section 6 and 7 predictions in `cache/` (not in the repo), so later runs take a few minutes; set `RERUN = True` in the first code cell to retrain them, for example after changing a feature or a model setting.

## About me

I'm Ashwin, looking for internships for summer 2027 in AI/ML, data science or computational chemistry. Feedback is very welcome.

- LinkedIn: [Ashwin Arunachalam](https://www.linkedin.com/in/ashwin-arunachalam-1b0b91345)
- GitHub: [Ash679-ux](https://github.com/Ash679-ux)