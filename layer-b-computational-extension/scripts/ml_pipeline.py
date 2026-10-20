"""
ml_pipeline.py
--------------
A small, honestly-labeled machine learning pipeline built on top of
data/simulated_respondent_data.csv.

WHAT THIS IS: a methodological demonstration of a standard supervised-
learning workflow (preprocessing -> encoding -> feature engineering ->
train/test split -> baseline -> models -> cross-validation -> evaluation
-> interpretation), built to show the workflow is understood end to end.

WHAT THIS IS NOT: evidence about Garissa County. The target variable
(`perceives_security_improving`) is synthetic -- see generate_data.py and
the thesis's own Appendix III for full disclosure. Model accuracy,
coefficients, and feature importances below describe patterns that were
DELIBERATELY built into the synthetic data generator, not patterns
discovered in real survey responses. Treat every number in this script's
output as "can I recover a known synthetic signal with this pipeline?",
not "what predicts security perceptions in Garissa?".

Models compared:
    0. Baseline  -- DummyClassifier (predicts the majority class)
    1. Logistic Regression -- interpretable linear baseline
    2. Decision Tree -- interpretable, captures nonlinear splits
    3. Random Forest -- ensemble, usually strongest of the three

Run (from repo root):
    python scripts/ml_pipeline.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "simulated_respondent_data.csv")
FIGDIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
NAVY, RUST, SAND, SAGE = "#1f3a5f", "#c0562b", "#d9a441", "#5b7f5e"
MODEL_COLORS = {"Baseline (majority class)": "#9a9a9a", "Logistic Regression": NAVY,
                 "Decision Tree": RUST, "Random Forest": SAGE}

RANDOM_STATE = 42

# =======================================================================
# 1. Load data
# =======================================================================
df = pd.read_csv(DATA)
TARGET = "perceives_security_improving"

# ---------------------------------------------------------------------
# 2. Feature engineering
#    - raw categoricals: gender, age_category, sub_county, primary_root_cause_theme
#    - one engineered feature: is_older (51+) -- collapses two age bands
#      into a single flag, since the synthetic generator applies a large,
#      similar penalty to both "51-65" and "Above 65".
# ---------------------------------------------------------------------
df["is_older_51_plus"] = df["age_category"].isin(["51-65", "Above 65"]).astype(int)

categorical_features = ["gender", "age_category", "sub_county", "primary_root_cause_theme"]
engineered_features = ["is_older_51_plus"]
all_features = categorical_features + engineered_features

X = df[all_features]
y = df[TARGET]

# ---------------------------------------------------------------------
# 3. Preprocessing pipeline
#    - missing-value handling: most-frequent-value imputation for
#      categoricals (defensive -- this synthetic dataset has none, but
#      the step is included so the pipeline is robust to real, messier
#      survey data with item non-response)
#    - categorical encoding: one-hot
# ---------------------------------------------------------------------
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_transformer, categorical_features),
    ("num", SimpleImputer(strategy="median"), engineered_features),
])

# ---------------------------------------------------------------------
# 4. Train/test split (stratified on the target, 75/25)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
print(f"Train target balance:\n{y_train.value_counts(normalize=True)}\n")

# ---------------------------------------------------------------------
# 5. Define models (baseline + 3 candidates), each wrapped with the
#    same preprocessing so comparisons are apples-to-apples.
# ---------------------------------------------------------------------
models = {
    "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE),
}

pipelines = {name: Pipeline([("prep", preprocessor), ("clf", clf)]) for name, clf in models.items()}

# =======================================================================
# Figure B0a -- class (target) distribution
# =======================================================================
class_counts = y.value_counts().rename({0: "No / Flat", 1: "Improving"})
fig, ax = plt.subplots(figsize=(5, 4.2))
bars = ax.bar(class_counts.index, class_counts.values, color=[RUST, NAVY])
for b in bars:
    ax.annotate(f"{int(b.get_height())} ({b.get_height()/len(df):.1%})",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Respondents")
ax.set_title("Figure B0a (SIMULATED DATA): Target Class Distribution\n"
              "perceives_security_improving (n=397)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figB0a_class_distribution.png"))
plt.close()
print(f"Class balance: {class_counts.to_dict()} -- a {class_counts.min()/class_counts.sum():.0%}/"
      f"{class_counts.max()/class_counts.sum():.0%} split, mild but real imbalance.")

# =======================================================================
# Figure B0b -- feature association heatmap (Cramer's V)
#    A Pearson correlation matrix isn't meaningful here -- every feature
#    is categorical/binary, not continuous. Cramer's V is the categorical
#    equivalent: a 0-1 measure of association between two categorical
#    variables, built on the same chi-square statistic used in Part 1.
# =======================================================================
from scipy.stats import chi2_contingency as _chi2c

def cramers_v(a, b):
    tab = pd.crosstab(a, b)
    chi2_, _, _, _ = _chi2c(tab.values)
    n_ = tab.values.sum()
    return np.sqrt(chi2_ / (n_ * (min(tab.shape) - 1)))

assoc_features = categorical_features + engineered_features + [TARGET]
assoc_matrix = pd.DataFrame(index=assoc_features, columns=assoc_features, dtype=float)
for f1 in assoc_features:
    for f2 in assoc_features:
        assoc_matrix.loc[f1, f2] = 1.0 if f1 == f2 else cramers_v(df[f1], df[f2])

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(assoc_matrix.values.astype(float), cmap="Blues", vmin=0, vmax=1)
labels = [f.replace("_", " ") for f in assoc_features]
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{assoc_matrix.values[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if assoc_matrix.values[i, j] > 0.5 else "black")
fig.colorbar(im, ax=ax, label="Cramer's V (0 = no association, 1 = perfect)")
ax.set_title("Figure B0b (SIMULATED DATA): Feature Association Matrix\n"
              "(Cramer's V -- the categorical equivalent of a correlation matrix)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figB0b_feature_association.png"))
plt.close()

# =======================================================================
# 6. 5-fold stratified cross-validation on the training set
# =======================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}
print("=" * 70)
print("5-FOLD CROSS-VALIDATION (on training set)")
print("=" * 70)
for name, pipe in pipelines.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv,
                             scoring=["accuracy", "f1", "roc_auc"])
    cv_results[name] = scores
    print(f"{name:28s}  acc={scores['test_accuracy'].mean():.3f}±{scores['test_accuracy'].std():.3f}  "
          f"f1={scores['test_f1'].mean():.3f}±{scores['test_f1'].std():.3f}  "
          f"roc_auc={scores['test_roc_auc'].mean():.3f}±{scores['test_roc_auc'].std():.3f}")

# =======================================================================
# Figure B6 -- cross-validation performance (mean +/- std across 5 folds)
# =======================================================================
cv_metrics = ["test_accuracy", "test_f1", "test_roc_auc"]
cv_labels = ["Accuracy", "F1", "ROC-AUC"]
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(cv_labels))
width = 0.2
for i, name in enumerate(pipelines.keys()):
    means = [cv_results[name][m].mean() for m in cv_metrics]
    stds = [cv_results[name][m].std() for m in cv_metrics]
    ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=3,
           label=name, color=MODEL_COLORS[name])
ax.set_xticks(x)
ax.set_xticklabels(cv_labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score (mean ± std, 5-fold stratified CV)")
ax.set_title("Figure B6 (SIMULATED DATA): 5-Fold Cross-Validation Performance\n"
              "(training set only -- generalization check before touching the test set)", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figB6_cv_performance.png"))
plt.close()

# =======================================================================
# 7. Fit on full training set, evaluate once on held-out test set
# =======================================================================
test_results = {}
print("\n" + "=" * 70)
print("HELD-OUT TEST SET PERFORMANCE (fit once, evaluated once)")
print("=" * 70)
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    test_results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "preds": preds,
        "proba": proba,
    }
    r = test_results[name]
    print(f"{name:28s}  acc={r['accuracy']:.3f}  prec={r['precision']:.3f}  "
          f"rec={r['recall']:.3f}  f1={r['f1']:.3f}  roc_auc={r['roc_auc']:.3f}")

# =======================================================================
# Figure B1 -- model comparison bar chart (test-set metrics)
# =======================================================================
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(metrics))
width = 0.2
for i, name in enumerate(pipelines.keys()):
    vals = [test_results[name][m] for m in metrics]
    ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=MODEL_COLORS[name])
ax.set_xticks(x)
ax.set_xticklabels([m.replace("_", " ").upper() for m in metrics])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score (held-out test set)")
ax.set_title("Figure B1 (SIMULATED DATA): Model Comparison on Held-Out Test Set")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figB1_model_comparison.png"))
plt.close()

# =======================================================================
# Figure B2 -- confusion matrices (3 candidate models, baseline excluded)
# =======================================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, name in zip(axes, ["Logistic Regression", "Decision Tree", "Random Forest"]):
    cm = confusion_matrix(y_test, test_results[name]["preds"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No/Flat", "Improving"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=10)
fig.suptitle("Figure B2 (SIMULATED DATA): Confusion Matrices, Held-Out Test Set", y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figB2_confusion_matrices.png"), bbox_inches="tight")
plt.close()

# =======================================================================
# Figure B3 -- ROC curves
# =======================================================================
fig, ax = plt.subplots(figsize=(6, 5.5))
for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
    fpr, tpr, _ = roc_curve(y_test, test_results[name]["proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={test_results[name]['roc_auc']:.2f})",
            color=MODEL_COLORS[name], linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Figure B3 (SIMULATED DATA): ROC Curves, Held-Out Test Set")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figB3_roc_curves.png"))
plt.close()

# =======================================================================
# Figure B4 -- Decision Tree structure (interpretation)
# =======================================================================
dt_pipe = pipelines["Decision Tree"]
feature_names = dt_pipe.named_steps["prep"].get_feature_names_out()
fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(dt_pipe.named_steps["clf"], feature_names=feature_names,
          class_names=["No/Flat", "Improving"], filled=True, rounded=True,
          fontsize=7, max_depth=3, ax=ax)
ax.set_title("Figure B4 (SIMULATED DATA): Decision Tree Structure (max depth shown = 3)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figB4_decision_tree.png"))
plt.close()

# =======================================================================
# Figure B5 -- Random Forest feature importances
# =======================================================================
rf_pipe = pipelines["Random Forest"]
rf_importances = pd.Series(
    rf_pipe.named_steps["clf"].feature_importances_, index=feature_names
).sort_values(ascending=True).tail(12)
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.barh(rf_importances.index.str.replace("cat__", "").str.replace("num__", ""),
        rf_importances.values, color=SAGE)
ax.set_xlabel("Gini importance")
ax.set_title("Figure B5 (SIMULATED DATA): Random Forest Feature Importances\n(top 12)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figB5_rf_feature_importance.png"))
plt.close()

# =======================================================================
# Summary table -> CSV for the README
# =======================================================================
summary = pd.DataFrame({name: {m: test_results[name][m] for m in metrics} for name in pipelines.keys()}).T
summary.to_csv(os.path.join(FIGDIR, "..", "data", "model_comparison_results.csv"))
print("\nSummary written to data/model_comparison_results.csv")
print(summary.round(3))

cv_summary = pd.DataFrame({
    name: {f"{lbl}_mean": cv_results[name][m].mean()
           for m, lbl in zip(cv_metrics, cv_labels)}
          | {f"{lbl}_std": cv_results[name][m].std()
             for m, lbl in zip(cv_metrics, cv_labels)}
    for name in pipelines.keys()
}).T
cv_summary.to_csv(os.path.join(FIGDIR, "..", "data", "cv_results.csv"))
print("\nCV summary written to data/cv_results.csv")
print(cv_summary.round(3))

print("\nAll figures written to:", FIGDIR)
