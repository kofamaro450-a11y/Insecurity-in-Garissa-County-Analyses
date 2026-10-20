# Layer B — Garissa Insecurity: A Computational Social Science Extension

This project extends my undergraduate research (Layer A) using reproducible
quantitative and machine-learning workflows. Part 1 rebuilds the thesis's own
real reported statistics. Part 2 is a synthetic-data ML pipeline
demonstration, necessary because respondent-level data from the original 2023
fieldwork was not retained. **Part 3 adds a real classification analysis on
independently-sourced, government-published data** (Kenya's National Crime
Research Centre), so the machine-learning component of this project is not
entirely synthetic.

Three things live in this layer, on three different kinds of footing:

| | Basis | What it shows |
|---|---|---|
| **Part 1 — Descriptive reproduction** (§1 below) | **Real**, reported thesis figures | I can take a real research finding and rebuild it as a reproducible, testable artifact |
| **Part 2 — ML pipeline demonstration** (§2 below) | **Synthetic** data, built to match the real margins | I can build, evaluate, and interpret a standard supervised-learning pipeline end to end |
| **Part 3 — Real cross-county classification** (§3 below) | **Real**, government-published NCRC crime-perception data, 16 Kenyan counties | The same pipeline, run on genuine data, with a real (if small) predictive result |

---

## Part 1 — Descriptive reproduction (real data)

Tables 4.1–4.4 and Figures 4.1–4.5 in the thesis report real, aggregated

descriptive statistics from the actual 397-respondent survey. `scripts/analysis.py`
reconstructs a respondent-level contingency table from those real reported
margins and runs a chi-square test of independence (age category × gender):

**χ²(3, N=397) = 23.91, p < .001, Cramér's V = 0.245** — this matches the value
reported in the thesis exactly, since it's built from the same real numbers.

| Figure | Content |
|---|---|
| `fig4_1_gender.png` | Gender split (59% female / 41% male) |
| `fig4_2_age.png` | Age distribution |
| `fig4_3_age_by_gender.png` | Age distribution by gender |
| `fig4_4_chi_square.png` | Observed vs. expected counts, chi-square test |
| `fig4_5_subcounty.png` | Sub-county distribution |

See `data/external_datasets.md` for real, public datasets (ACLED, UNHCR, KNBS,
GTD) that a future study could use to replace Part 2's synthetic demonstration
with a genuine empirical model.

## Part 2 — ML pipeline demonstration (synthetic data)

**Target variable:** `perceives_security_improving` — a synthetic binary outcome.
The thesis's own Appendix III states plainly that this field was generated
specifically to give a modeling pipeline something non-trivial to work on. It is
not a measured survey response. **Model accuracy, coefficients, and feature
importances below describe how well each model recovers a known synthetic
signal — not what predicts security perceptions in Garissa.**

### Pipeline (`scripts/ml_pipeline.py`)

```
Raw data
  → Missing-value handling      (most-frequent imputation for categoricals;
                                  defensive — this synthetic set has no
                                  missingness, but real survey data would)
  → Categorical encoding        (one-hot: gender, age_category, sub_county,
                                  primary_root_cause_theme)
  → Feature engineering         (is_older_51_plus: collapses two age bands
                                  the data-generating process treats similarly)
  → Train/test split            (75/25, stratified on the target)
  → Baseline                    (DummyClassifier, predicts majority class)
  → Models                      (Logistic Regression, Decision Tree, Random Forest)
  → 5-fold stratified CV        (on the training set only)
  → Evaluation                  (held-out test set: accuracy, precision, recall,
                                  F1, ROC-AUC, confusion matrix, ROC curve)
  → Interpretation              (LR coefficients, DT structure, RF feature importance,
                                  feature association matrix, consistency across models)
```

### Models compared

| Model | Why it's here |
|---|---|
| Baseline (majority class) | Floor for comparison — any real model should beat this |
| Logistic Regression | Interpretable linear baseline (also used standalone in Appendix III) |
| Decision Tree | Interpretable, captures nonlinear/interaction splits |
| Random Forest | Ensemble method, usually the strongest of the three |

Three models, not ten — each one is fully interpreted below rather than left as
a line in a leaderboard.

### Model evaluation (held-out test set, n=100)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
