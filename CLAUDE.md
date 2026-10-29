# CLAUDE.md — IntelliGrid: Electricity Theft and Anomaly Detection

College Data Science project. The submitted report is
`IntelliGrid Electricity Theft and Anomaly Detection (1).pdf` (repo root, 6 pages).
**The report is the specification.** Follow its architecture; do not add models,
features or steps it does not describe unless the team approves them first.

---

## 1. Development rules (strict)

### Code style
- All ML/DL work lives in **Jupyter notebooks (`.ipynb`)**. No `.py` files containing
  project logic, and no helper modules imported by the notebooks.
- Code is written **sequentially, cell by cell**, top to bottom. A notebook must run
  cleanly with "Restart & Run All".
- **No `def`**: no custom Python functions.
- **No `lambda`**: this includes `.apply(lambda ...)`, `key=lambda ...` and lambdas
  passed to `FunctionTransformer`.
- **No custom classes**: no `class`, no Keras model/layer subclassing, and no custom
  loss, metric or callback classes.
- Allowed instead: vectorised Pandas/NumPy operations, `for` loops, `np.where`,
  boolean masks, built-in aggregation names (`.agg(["mean", "std"])`),
  scikit-learn built-in estimators/transformers/`Pipeline`, and the Keras
  `Sequential` or Functional API with built-in layers, losses, metrics and callbacks
  (`EarlyStopping`, `ModelCheckpoint`). Passing an existing library function is fine
  (for example `FunctionTransformer(np.log1p)`), because nothing new is being defined.
- Do not use `ipywidgets.interact` or `.observe` in the demo. Both need a callback
  function. Notebook 08 uses an editable parameter cell (`CUSTOMER_ID = ...`) that
  the user changes before re-running the cells below it.
- Libraries: Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, plus joblib
  (saving models), pyarrow (parquet) and SHAP (Notebook 07). Seaborn is optional.
  Target environment: **Google Colab**.
- Audience: undergraduate Data Science students. Prefer clear and explicit over
  clever. Put a short markdown cell before each logical step that explains *what*
  the step does and *why*. Use descriptive variable names and avoid long chained
  one-liners.
- Every notebook starts with: (1) a markdown title and purpose cell, (2) an imports
  cell, (3) a config cell (paths, `RANDOM_STATE = 42`, constants). Set seeds with
  `np.random.seed(RANDOM_STATE)` and `tf.keras.utils.set_random_seed(RANDOM_STATE)`.

### Data leakage prevention
- **Split by customer, before anything is fitted.** Use one shuffled, stratified
  (on `FLAG`) train/validation/test split with a fixed seed. Create it once in
  Notebook 02, save the customer IDs, and reuse those exact IDs in every later
  notebook.
- **Row-wise operations** (they use only one customer's own readings) may be applied
  to all customers: sorting dates, setting the 3 Oct 2014 glitch to NaN, per-customer
  capping, per-customer interpolation, and per-customer feature calculation.
- **Any statistic computed across customers** must be fitted on **training customers
  only** and then applied to validation and test. This covers imputers, scalers,
  global caps or percentiles, and class weights.
- **The theft customers are stored first in the raw file.** Never use Keras
  `validation_split` (it takes the last rows without shuffling). Always pass
  `validation_data=(X_val, y_val)` explicitly.
- **The validation set** is used for early stopping, hyperparameter choices and
  **all thresholds**.
- **The test set is used exactly once**, in Notebook 06. Notebooks 04 and 05 may
  report validation metrics only. Nothing learned in Notebooks 07 or 08 may feed back
  into models or thresholds.
- **Unsupervised models never train on labels.** Labels are used to evaluate them.
  The only label use in their training is the report's rule that the autoencoder
  learns from *normal training customers only*.
- Feature definitions and constants (for example the spike multiplier) are fixed by
  reasoning or validation data, never tuned on test results.

### Results integrity
- **Never fabricate, estimate or "fill in" results.** Every number in a markdown cell,
  README or summary must come from an executed cell output in the committed notebook,
  or from a results file that a notebook wrote.
- If a cell fails, a model underperforms or a run was not executed, say so plainly.
- Commit notebooks **with their outputs**, so every reported number can be traced.
- Metric tables are written to disk by code (`reports/results/*.csv`), not typed by
  hand.
- The figures in the report (for example ROC-AUC 0.66 for average consumption, or the
  missing-value percentages) are the team's earlier findings. Re-derive them in the
  notebooks; do not copy them in as results.

---

## 2. Dataset (SGCC) — facts stated in the report

| Property | Value (report, "verified from the file") |
|---|---|
| Source | State Grid Corporation of China, released with Zheng et al., IEEE TII 2018 |
| Customers | 42,372 rows, one per customer, no duplicates |
| Labels | 3,615 theft (8.5%), 38,757 normal. All-normal accuracy = 91.5% |
| Period | 1 Jan 2014 – 31 Oct 2016, 1,034 daily columns (18 Sep 2016 absent) |
| Missing | 25.6% overall; 31.5% theft vs 25.1% normal; ~36% in 2014–15, under 2% in 2016 |
| Typical usage | Median 4.6 kWh/customer/day |

Known traps, to be handled in Notebooks 01–02:
1. Date columns are in **text order** (`2014/1/1, 2014/1/10, 2014/1/11, …`). Parse them
   and sort them chronologically.
2. **All theft customers come first.** Shuffle before splitting and training.
3. **System-wide glitch on 3 Oct 2014**: the median reading jumps from 4.5 to 144 kWh.
   Set this day to NaN for every customer.
4. **Extreme readings** reach 800,003 kWh in a single day. Cap them.
5. **Missingness is time-dependent** because meters came online over time.
6. **The weekly cycle is weak** and theft customers are no less regular than normal
   ones. Do not claim that theft "breaks a weekly rhythm".
7. **Baseline floor**: average consumption alone gives ROC-AUC of about 0.66. Every
   model must clearly beat this. Recompute it on our test split in Notebook 06.

Expected raw-file layout (confirm in Notebook 01): a customer ID column (`CONS_NO`),
a label column (`FLAG`, 1 = theft) and one column per date. The raw file is **not**
in the repository and must not be committed because of its size. It goes in
`data/raw/`.

Limitations to keep in mind (from the report): labels are per customer, with no theft
date and no documented labelling method; the data covers one utility in one region.
A flag is a lead for human investigation, **not proof of theft**.

---

## 3. Architecture (from the report)

```
1 Data preparation & customer split   (sort dates, fix glitch day, cap outliers, impute gaps)
      Train: fits all models | Validation: tuning + thresholds | Test: final evaluation only
            |                                              |
2 Supervised branch (uses labels)            3 Unsupervised branch (no labels)
  * LSTM classifier — weekly sequence          * Dense autoencoder — normal train customers only
  - Logistic Regression, Random Forest         - Isolation Forest — engineered features
    (engineered features)
            |                                              |
      Theft probability                       Anomaly score (reconstruction error)
            \______________________________________________/
4 Evaluation — same held-out test customers, thresholds set on validation
5 Investigation view (notebook demo) — consumption history, theft probability,
  anomaly score, unusual periods
(* = deep learning models; the others are baselines)
```

### The five models
| # | Model | Type | Input | Question it answers |
|---|---|---|---|---|
| 1 | Logistic Regression | Supervised, classical | Engineered features | Simplest baseline |
| 2 | Random Forest | Supervised, classical | Engineered features | Strong non-deep baseline + feature importance |
| 3 | LSTM classifier | Supervised, deep | Weekly consumption sequence | Does learning from the sequence beat hand-made features? |
| 4 | Isolation Forest | Unsupervised, classical | Engineered features | Anomaly detection without labels or deep learning |
| 5 | Dense autoencoder | Unsupervised, deep | Consumption vector, trained on normal train customers | Does learned "normal" flag theft better than Isolation Forest? |

Supervised models are compared with supervised models, and unsupervised models with
unsupervised models.

### Metrics (Notebook 06)
Precision, recall, F1, ROC-AUC, PR-AUC, false-positive rate, confusion matrix, and
**precision among the top-ranked customers** (precision@k, the inspection-budget view).
Accuracy is not used as a headline metric.

---

## 4. Eight-notebook workflow and file contract

Notebooks live in `notebooks/` and run in order. Each one saves exactly what the next
one needs.

| NB | File | Does | Reads | Writes |
|---|---|---|---|---|
| 01 | `01_data_understanding_eda.ipynb` | Load, sort dates, profile missing values / zeros / outliers / imbalance / trends, glitch day, baseline floor (exploratory only) | `data/raw/` | `reports/figures/01_*` (no modelling artefacts) |
| 02 | `02_preprocessing.ipynb` | Fix column order (insert 18 Sep 2016 as NaN), glitch → NaN, cap, stratified customer split, interpolate short gaps, train-fitted imputer, weekly aggregation, train-fitted scaling | raw file | `data/processed/` cleaned daily matrix, split IDs, weekly sequences per split, fitted imputer/scaler (`models/preprocessing/`) |
| 03 | `03_feature_engineering.ipynb` | Per-customer features: mean, spread, zero-day ratio, longest zero run, sudden drops and spikes, volatility, change from own baseline | NB02 outputs | `data/processed/features_{train,val,test}.parquet` |
| 04 | `04_classical_baselines.ipynb` | LR + RF (supervised), Isolation Forest (unsupervised) on features; validation metrics only | NB03 features | `models/*.joblib`, validation scores |
| 05 | `05_deep_learning_models.ipynb` | LSTM on weekly sequences (labels); dense autoencoder on normal training customers; validation metrics only | NB02 sequences | `models/*.keras`, training histories, validation scores |
| 06 | `06_evaluation.ipynb` | Choose thresholds on validation (one rule per group), then score test **once**; all metrics + plots | all models, val/test data | `reports/results/*.csv`, test scores, figures |
| 07 | `07_explainability_risk.ipynb` | RF feature importance + SHAP; autoencoder reconstruction-error breakdown by period; flagged customers' histories | saved models + scores | figures, risk table |
| 08 | `08_interactive_demo.ipynb` | Investigation view: choose a test customer → consumption, theft probability, anomaly score, unusual periods (saved models only, no retraining) | saved models + scores | none |

Planned repository layout:
```
notebooks/            01_… to 08_… .ipynb
data/raw/             SGCC file (git-ignored)
data/processed/       intermediate arrays/parquet (git-ignored)
models/               saved models, imputer, scaler (git-ignored unless small)
reports/figures/      PNG figures
reports/results/      metric CSVs written by notebooks
CLAUDE.md, README.md, requirements.txt, .gitignore
```

---

## 5. Working agreement
- Build **one notebook per phase**, in order. Stop after each notebook for team review
  and approval before starting the next one.
- Do not change an approved notebook's outputs or contract without saying so; later
  notebooks depend on them.
- Design choices the report leaves open (split ratio, capping rule, imputation for
  long gaps, scaling, precision@k cut-offs, threshold rule) are proposed and approved
  before implementation, and then recorded in the "Decisions log" below.

## 6. Decisions log
_(Empty. Add each approved decision here with its date.)_

## 7. Status
- Present in the repository: the report PDF only.
- Missing: dataset (to be placed in `data/raw/`), all 8 notebooks (the report says
  early Notebook 01 cells were written, but they are not in the repository),
  `README.md`, `requirements.txt`, `.gitignore`, folder structure.
