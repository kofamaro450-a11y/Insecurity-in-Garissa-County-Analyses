"""
analysis.py
-----------
Reproduces every quantitative figure and statistical test in the thesis
(Chapter Four + Appendix III) from data/simulated_respondent_data.csv.

Tools: pandas (data handling), matplotlib (charts), scipy.stats
(chi-square test of independence), scikit-learn (logistic regression).

Run (from repo root):
    python scripts/analysis.py
Produces PNG figures in figures/ and prints a text summary of every
statistic to the console (redirect to figures/analysis_log.txt if wanted).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "simulated_respondent_data.csv")
FIGDIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
NAVY = "#1f3a5f"
RUST = "#c0562b"
SAND = "#d9a441"
SAGE = "#5b7f5e"
PALETTE = [NAVY, RUST, SAND, SAGE, "#7a6fa6", "#3f8f8a", "#a54848", "#5a5a5a"]

df = pd.read_csv(DATA)
AGE_ORDER = ["18-35", "36-50", "51-65", "Above 65"]
SUBCOUNTY_ORDER = ["Fafi", "Garissa", "Lagdera", "Dadaab", "Balambala"]

# =======================================================================
# Figure 4.1 -- Gender of respondents
# =======================================================================
gender_counts = df["gender"].value_counts()
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(
    gender_counts.values,
    labels=[f"{g}\n{c} ({c/len(df):.0%})" for g, c in gender_counts.items()],
    colors=[NAVY, RUST],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
ax.set_title("Figure 4.1: Gender of Respondents (n = 397)")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig4_1_gender.png"))
plt.close()

# =======================================================================
# Figure 4.2 -- Age distribution
# =======================================================================
age_counts = df["age_category"].value_counts().reindex(AGE_ORDER)
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(age_counts.index, age_counts.values, color=NAVY)
for b in bars:
    ax.annotate(f"{int(b.get_height())} ({b.get_height()/len(df):.1%})",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Number of respondents")
ax.set_title("Figure 4.2: Age Distribution of Respondents (n = 397)")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig4_2_age.png"))
plt.close()

# =======================================================================
# Figure 4.3 -- Age distribution by gender (grouped bars, % within gender)
# =======================================================================
ct = pd.crosstab(df["age_category"], df["gender"]).reindex(AGE_ORDER)
ct_pct = ct.div(ct.sum(axis=0), axis=1) * 100
fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(len(AGE_ORDER))
width = 0.35
ax.bar(x - width / 2, ct_pct["Male"], width, label="Male", color=NAVY)
ax.bar(x + width / 2, ct_pct["Female"], width, label="Female", color=RUST)
ax.set_xticks(x)
ax.set_xticklabels(AGE_ORDER)
ax.set_ylabel("% within gender")
ax.set_title("Figure 4.3: Age Distribution by Gender")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig4_3_age_by_gender.png"))
plt.close()

# =======================================================================
# 4.2.3 -- Chi-square test of independence: age category x gender
# =======================================================================
chi2, p, dof, expected = chi2_contingency(ct.values)
n = ct.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

print("=" * 70)
print("CHI-SQUARE TEST OF INDEPENDENCE: age_category x gender")
print("=" * 70)
print("Observed contingency table:\n", ct)
print(f"\nchi2 = {chi2:.2f}, dof = {dof}, p = {p:.6f}")
print(f"Cramer's V = {cramers_v:.3f}")
print(f"Critical value (alpha=0.05, dof={dof}): {7.815 if dof==3 else 'see chi2 table'}")

# Figure 4.4 -- Observed vs expected counts
expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(AGE_ORDER))
width = 0.2
ax.bar(x - 1.5 * width, ct["Male"], width, label="Observed - Male", color=NAVY)
ax.bar(x - 0.5 * width, expected_df["Male"], width, label="Expected - Male", color=NAVY, alpha=0.4, hatch="//")
ax.bar(x + 0.5 * width, ct["Female"], width, label="Observed - Female", color=RUST)
ax.bar(x + 1.5 * width, expected_df["Female"], width, label="Expected - Female", color=RUST, alpha=0.4, hatch="//")
ax.set_xticks(x)
ax.set_xticklabels(AGE_ORDER)
ax.set_ylabel("Count")
ax.set_title(f"Figure 4.4: Observed vs. Expected Counts under Independence\n"
             f"χ²({dof}, N={n}) = {chi2:.2f}, p < .001, Cramér's V = {cramers_v:.3f}")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig4_4_chi_square.png"))
plt.close()

# =======================================================================
# Figure 4.5 -- Sub-county distribution
# =======================================================================
sc_counts = df["sub_county"].value_counts().reindex(SUBCOUNTY_ORDER)
fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.bar(sc_counts.index, sc_counts.values, color=SAGE)
for b in bars:
    ax.annotate(f"{int(b.get_height())} ({b.get_height()/len(df):.1%})",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=9)
ax.axhline(len(df) / 5, color=RUST, linestyle="--", linewidth=1, label="Equal-share target (79.4)")
ax.set_ylabel("Number of respondents")
ax.set_title("Figure 4.5: Distribution of Respondents by Sub-County (n = 397)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig4_5_subcounty.png"))
plt.close()

# =======================================================================
# Figure A2 -- Thematic frequency coding (SIMULATED)
# =======================================================================
theme_counts = df["primary_root_cause_theme"].value_counts().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
bars = ax.barh(theme_counts.index, theme_counts.values, color=PALETTE[:len(theme_counts)])
for b in bars:
    ax.annotate(f"{int(b.get_width())} ({b.get_width()/len(df):.1%})",
                (b.get_width(), b.get_y() + b.get_height() / 2),
                ha="left", va="center", fontsize=8, xytext=(3, 0), textcoords="offset points")
ax.set_xlabel("Number of respondents (SIMULATED)")
ax.set_title("Figure A2 (SIMULATED DATA): Illustrative Coded\nRoot-Cause Theme Frequencies", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figA2_theme_frequency.png"))
plt.close()

# =======================================================================
# Figure A3 -- Logistic regression coefficients (SIMULATED)
# =======================================================================
X_cat = df[["age_category", "sub_county", "gender"]].copy()
enc = OneHotEncoder(drop=None, sparse_output=False)
X_enc = enc.fit_transform(X_cat)
feature_names = enc.get_feature_names_out(["age_category", "sub_county", "gender"])
y = df["perceives_security_improving"].values

# Manually drop reference categories: age_18-35, sub_county_Balambala, gender_Female
ref_drop = ["age_category_18-35", "sub_county_Balambala", "gender_Female"]
keep_idx = [i for i, f in enumerate(feature_names) if f not in ref_drop]
X_final = X_enc[:, keep_idx]
kept_names = [feature_names[i] for i in keep_idx]

clf = LogisticRegression(max_iter=1000)
clf.fit(X_final, y)
coefs = pd.Series(clf.coef_[0], index=kept_names).sort_values()

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION (SIMULATED OUTCOME): perceives_security_improving")
print("Reference categories: age 18-35, sub-county Balambala, gender Female")
print("=" * 70)
print(coefs.to_string())
print(f"\nIntercept (log-odds): {clf.intercept_[0]:.3f}")
print(f"Training accuracy: {clf.score(X_final, y):.3f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = [RUST if v < 0 else NAVY for v in coefs.values]
ax.barh(coefs.index.str.replace("age_category_", "Age: ")
                     .str.replace("sub_county_", "Sub-county: ")
                     .str.replace("gender_", "Gender: "),
        coefs.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient (log-odds)")
ax.set_title("Figure A3 (SIMULATED DATA): Logistic Regression Coefficients\n"
             "Ref.: Age 18-35, Sub-County Balambala, Gender Female", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "figA3_logistic_coefficients.png"))
plt.close()

print("\nAll figures written to:", FIGDIR)
