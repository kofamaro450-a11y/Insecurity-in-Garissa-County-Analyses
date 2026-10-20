"""
generate_data.py
-----------------
Builds the SIMULATED respondent-level dataset used in Appendix III of the
thesis "Assessment of Kenya's Current Insecurity Challenges: A Case Study
of Security Challenges in Garissa County" (portfolio edition, 2026).

IMPORTANT: This is a synthetic dataset. Respondent-level data from the
original 2023 fieldwork was not retained -- only the aggregated tables
in Chapter Four (Tables 4.1-4.4, Figures 4.1-4.5) survive. This script
reconstructs a respondent-level file that reproduces those exact reported
margins for gender and age category (Table 4.3), allocates sub-county as
evenly as the quota-sampling design intended, and adds two illustrative
fields (root-cause theme, perceived security trend) so that thematic-
frequency counting and logistic regression have something non-trivial to
work on. Those two fields are NOT real survey responses -- see README.md
and the thesis Appendix III for full disclosure.

Run (from repo root):
    python scripts/generate_data.py
Produces:
    data/simulated_respondent_data.csv  (397 rows)
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "..", "data", "simulated_respondent_data.csv")

# ---------------------------------------------------------------------
# 1. Gender x Age category -- reproduces Table 4.3 exactly (n = 397)
#    (Table 4.3 uses the counts as captioned in the original document:
#    Male n=234, Female n=163 -- see the thesis's own data-quality note
#    on the Fig 4.1 / Table 4.3 label swap.)
# ---------------------------------------------------------------------
age_gender_counts = {
    ("Male", "18-35"): 20,
    ("Male", "36-50"): 137,
    ("Male", "51-65"): 40,
    ("Male", "Above 65"): 37,
    ("Female", "18-35"): 21,
    ("Female", "36-50"): 92,
    ("Female", "51-65"): 46,
    ("Female", "Above 65"): 4,
}
assert sum(age_gender_counts.values()) == 397

rows = []
respondent_id = 1
for (gender, age_cat), n in age_gender_counts.items():
    for _ in range(n):
        rows.append({"respondent_id": respondent_id, "gender": gender, "age_category": age_cat})
        respondent_id += 1

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------
# 2. Sub-county -- Table 4.4 in the source document is internally
#    inconsistent (100 + 99*4 = 496, not 397; see README "Known data
#    issues"). We instead allocate as evenly as possible across the five
#    sub-counties so the total matches n = 397, consistent with the
#    quota-sampling narrative of an ~equal 20% target share per sub-county.
# ---------------------------------------------------------------------
subcounties = ["Fafi", "Garissa", "Lagdera", "Dadaab", "Balambala"]
base, remainder = divmod(397, 5)
counts = [base + 1 if i < remainder else base for i in range(5)]  # [80,80,79,79,79]
RNG.shuffle(counts)
subcounty_assignment = np.concatenate([[sc] * c for sc, c in zip(subcounties, counts)])
RNG.shuffle(subcounty_assignment)
df["sub_county"] = subcounty_assignment

# ---------------------------------------------------------------------
# 3. SIMULATED root-cause theme (section 4.3.1's 8 themes) -- illustrative
#    distribution only, not measured.
# ---------------------------------------------------------------------
themes = [
    "Ethnic & Clan Conflicts",
    "Border Issues",
    "Poverty & Unemployment",
    "Marginalization & Inequality",
    "Weak Governance",
    "Political Instability",
    "Proximity to Conflict Zones",
    "Remote & Ungoverned Areas",
]
theme_probs = [0.20, 0.16, 0.18, 0.13, 0.12, 0.08, 0.08, 0.05]
df["primary_root_cause_theme"] = RNG.choice(themes, size=len(df), p=theme_probs)

# ---------------------------------------------------------------------
# 4. SIMULATED binary outcome: "perceives security as improving"
#    Built with a loose, illustrative logistic structure in age and
#    sub-county (per Appendix A.3), purely so the regression demo has a
#    non-trivial pattern to recover. NOT a real measured outcome.
# ---------------------------------------------------------------------
age_effect = {"18-35": 0.0, "36-50": -0.1, "51-65": -0.9, "Above 65": -0.3}
subcounty_effect = {"Balambala": 0.0, "Garissa": 0.7, "Lagdera": 0.5, "Dadaab": 0.1, "Fafi": -0.8}
gender_effect = {"Female": 0.0, "Male": 0.1}

logit = (
    -0.2
    + df["age_category"].map(age_effect)
    + df["sub_county"].map(subcounty_effect)
    + df["gender"].map(gender_effect)
    + RNG.normal(0, 0.6, size=len(df))  # noise
)
prob = 1 / (1 + np.exp(-logit))
df["perceives_security_improving"] = (RNG.uniform(size=len(df)) < prob).astype(int)

# Shuffle row order so it doesn't read as sorted-by-construction
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["respondent_id"] = range(1, len(df) + 1)

df.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(df)} rows to {OUT_PATH}")
print(df.head())
print("\nGender counts:\n", df["gender"].value_counts())
print("\nAge counts:\n", df["age_category"].value_counts())
print("\nSub-county counts:\n", df["sub_county"].value_counts())
