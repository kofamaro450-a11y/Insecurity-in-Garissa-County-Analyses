"""
real_data_pipeline.py
----------------------
A REAL classification analysis using government-published data: can a
county's crime-perception profile (from Kenya's National Crime Research
Centre 2020 County Crime survey) predict whether that county belongs to
Kenya's historically-recognized arid/frontier region?

Data: 16 Kenyan counties (10 ASAL/frontier incl. Garissa + 6 non-frontier
comparators), 14 crime-perception percentages each, manually transcribed
from https://www.crimeresearch.go.ke/<county>/ (2020 survey wave).

Target (is_asal_frontier): a REAL, externally-defined classification --
Kenya's "hardcore arid" ASAL/frontier counties per the National Policy for
the Sustainable Development of Northern Kenya and other Arid Lands
(Sessional Paper No. 8 of 2012) -- NOT derived from the crime data itself.

Why Leave-One-Out CV, not a train/test split: with only 16 rows, a single
train/test split would leave ~4 test rows -- too few for a stable estimate.
Leave-One-Out Cross-Validation (LOOCV) uses every county as a test case
exactly once, which is the standard choice for genuinely small real datasets.

Run (from repo root):
    python scripts/real_data_pipeline.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "kenya_county_crime_perceptions_2020.csv")
FIGDIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
NAVY, RUST, SAND, SAGE = "#1f3a5f", "#c0562b", "#d9a441", "#5b7f5e"
RANDOM_STATE = 42

df = pd.read_csv(DATA)
CRIME_COLS = [c for c in df.columns if c not in ("county", "is_asal_frontier")]
X = df[CRIME_COLS]
y = df["is_asal_frontier"]

print(f"Dataset: {len(df)} counties, {len(CRIME_COLS)} crime-perception features")
print(f"Class balance: {y.value_counts().to_dict()} (1=ASAL/frontier, 0=non-frontier)")
print(f"Missing values: {X.isna().sum().sum()} cells, imputed with column medians\n")

# =======================================================================
# Figure C1 -- Group comparison: mean crime-perception by frontier status
# =======================================================================
group_means = df.groupby("is_asal_frontier")[CRIME_COLS].mean().T
group_means.columns = ["Non-frontier (n=6)", "ASAL/Frontier (n=10)"]
group_means = group_means.reindex(group_means["ASAL/Frontier (n=10)"].sub(group_means["Non-frontier (n=6)"]).sort_values().index)

fig, ax = plt.subplots(figsize=(8, 6.5))
y_pos = np.arange(len(group_means))
ax.barh(y_pos - 0.2, group_means["Non-frontier (n=6)"], height=0.4, label="Non-frontier (n=6)", color=NAVY)
ax.barh(y_pos + 0.2, group_means["ASAL/Frontier (n=10)"], height=0.4, label="ASAL/Frontier (n=10)", color=RUST)
ax.set_yticks(y_pos)
ax.set_yticklabels([c.replace("_", " ").title() for c in group_means.index], fontsize=8)
ax.set_xlabel("Mean % of respondents perceiving crime type as prevalent")
ax.set_title("Figure C1 (REAL DATA, NCRC 2020): Crime Perception by\nASAL/Frontier vs. Non-Frontier County Status", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figC1_group_comparison.png"))
plt.close()

# =======================================================================
# Independent-samples t-tests: which crime types differ most by group?
# =======================================================================
print("=" * 70)
print("INDEPENDENT-SAMPLES T-TESTS: ASAL/frontier vs non-frontier counties")
print("=" * 70)
results = []
for col in CRIME_COLS:
    a = df.loc[df.is_asal_frontier == 1, col].dropna()
    b = df.loc[df.is_asal_frontier == 0, col].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    results.append((col, a.mean(), b.mean(), t, p))
results_df = pd.DataFrame(results, columns=["crime", "asal_mean", "nonasal_mean", "t", "p"]).sort_values("p")
print(results_df.round(3).to_string(index=False))
print("\n(n=16 counties total -- these are suggestive, not confirmatory; no multiple-comparison")
print("correction applied given the small, exploratory sample. Murder and theft-of-stock/cattle")
print("rustling show the largest, most plausible gaps, consistent with the thesis's own themes.)")

# =======================================================================
# Pipeline definition
# =======================================================================
preprocessor = Pipeline([("impute", SimpleImputer(strategy="median"))])

models = {
    "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": Pipeline([("prep", preprocessor), ("scale", StandardScaler()),
                                       ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    "Decision Tree": Pipeline([("prep", preprocessor),
                                 ("clf", DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=RANDOM_STATE))]),
    "Random Forest": Pipeline([("prep", preprocessor),
                                 ("clf", RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=RANDOM_STATE))]),
}
MODEL_COLORS = {"Baseline (majority class)": "#9a9a9a", "Logistic Regression": NAVY,
                 "Decision Tree": RUST, "Random Forest": SAGE}

# Baseline needs the same imputed X (it ignores features but sklearn still wants no-NaN)
X_imputed = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=CRIME_COLS)

# =======================================================================
# Leave-One-Out Cross-Validation
# =======================================================================
loo = LeaveOneOut()
print("\n" + "=" * 70)
print("LEAVE-ONE-OUT CROSS-VALIDATION (each of 16 counties held out once)")
print("=" * 70)
loo_results = {}
for name, model in models.items():
    Xin = X_imputed if name == "Baseline (majority class)" else X
    preds = cross_val_predict(model, Xin, y, cv=loo)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    loo_results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "preds": preds}
    print(f"{name:28s}  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")

# =======================================================================
# Figure C2 -- LOOCV performance comparison
# =======================================================================
metrics = ["accuracy", "precision", "recall", "f1"]
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(metrics))
width = 0.2
for i, name in enumerate(models.keys()):
    vals = [loo_results[name][m] for m in metrics]
    ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=MODEL_COLORS[name])
ax.set_xticks(x)
ax.set_xticklabels([m.title() for m in metrics])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score (Leave-One-Out CV)")
ax.set_title("Figure C2 (REAL DATA): Model Comparison, Leave-One-Out CV\n(n=16 counties -- every county held out exactly once)", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figC2_loocv_comparison.png"))
plt.close()

# =======================================================================
# Figure C3 -- Confusion matrices (candidate models)
# =======================================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, name in zip(axes, ["Logistic Regression", "Decision Tree", "Random Forest"]):
    cm = confusion_matrix(y, loo_results[name]["preds"])
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-frontier", "ASAL/Frontier"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=10)
fig.suptitle("Figure C3 (REAL DATA): Confusion Matrices, Leave-One-Out CV", y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figC3_loocv_confusion.png"), bbox_inches="tight")
plt.close()

# =======================================================================
# Figure C4 -- Random Forest feature importance (fit on all 16 counties)
# =======================================================================
rf_final = Pipeline([("prep", preprocessor),
                       ("clf", RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=RANDOM_STATE))])
rf_final.fit(X, y)
importances = pd.Series(rf_final.named_steps["clf"].feature_importances_, index=CRIME_COLS).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.barh([c.replace("_", " ").title() for c in importances.index], importances.values, color=SAGE)
ax.set_xlabel("Gini importance")
ax.set_title("Figure C4 (REAL DATA): Random Forest Feature Importance\n(fit on all 16 counties)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figC4_rf_importance.png"))
plt.close()

# =======================================================================
# Figure C5 -- Decision tree structure (fit on all 16 counties)
# =======================================================================
dt_final = Pipeline([("prep", preprocessor),
                       ("clf", DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=RANDOM_STATE))])
dt_final.fit(X, y)
fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(dt_final.named_steps["clf"], feature_names=[c.replace("_", " ") for c in CRIME_COLS],
          class_names=["Non-frontier", "ASAL/Frontier"], filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title("Figure C5 (REAL DATA): Decision Tree Structure (fit on all 16 counties)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figC5_decision_tree.png"))
plt.close()

print("\nAll real-data figures written to:", FIGDIR)
