# Insecurity-in-Garissa-County-Analyses
Computational extension of my BA thesis on insecurity in Garissa County, Kenya. Not a repeat of the research: a reproducible layer on top of it, with chi-square analysis of the thesis findings, a synthetic-data ML demo, and a real classification on Kenya NCRC crime-perception data.
# Garissa Insecurity: Research + Computational Extension

*One research project from my research paper, presented in two layers.*
Ziebi Kofa Maro · BA Political Science, The Catholic University of Eastern Africa

## Start here

**[`Garissa_Insecurity_Analysis.ipynb`](Garissa_Insecurity_Analysis.ipynb)** is
the single, self-contained entry point to this project — GitHub renders it
with every chart and result already embedded, so it opens fully formed with
no setup required. If you'd rather not open a notebook viewer, the identical
content is also in **[`Garissa_Insecurity_Analysis.html`](Garissa_Insecurity_Analysis.html)**
(download and open in any browser). Everything below the two-layer summary is
supporting detail for anyone who wants to dig into the code, the raw data, or
the original thesis directly.

## Key results at a glance

**Part 1 (real data):** the sample was 59% female, concentrated in the 36–50
age band (57.4%), with age distribution differing significantly by gender
(χ²(3, N=397) = 23.91, p < .001, Cramér's V = 0.245).

**Part 2 (synthetic-data ML demonstration, 5-fold CV):**

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Baseline (majority class) | 0.58 | 0.00 | 0.50 |
| Logistic Regression | 0.60 | 0.50 | 0.64 |
| Decision Tree | 0.57 | 0.34 | 0.64 |
| Random Forest | 0.68 | 0.60 | 0.68 |

**Part 3 (real data — Kenya National Crime Research Centre, 16 counties,
Leave-One-Out CV):**

| Model | Accuracy | F1 |
|---|---|---|
| Baseline (majority class) | 0.625 | 0.769 |
| **Logistic Regression** | **0.875** | **0.909** |
| Decision Tree | 0.438 | 0.571 |
| Random Forest | 0.688 | 0.783 |

Both Part 2 and Part 3 show the same honest pattern: a simple linear model
generalizes better than a tree-based model at small sample sizes, and
cross-validation is what catches that rather than letting one favorable split
tell a misleading story. Part 3's data and target are entirely real; see the
notebook for the full discussion.

## Structure

```
├── Garissa_Insecurity_Analysis.ipynb      <- start here (notebook, GitHub-rendered)
├── Garissa_Insecurity_Analysis.html       <- same content, static HTML
│
├── layer-a-original-research/
│   ├── README.md
│   └── Garissa_Insecurity_Thesis_Portfolio.docx
│
└── layer-b-computational-extension/
    ├── README.md
    ├── data/
    │   ├── simulated_respondent_data.csv
    │   ├── kenya_county_crime_perceptions_2020.csv
    │   ├── external_datasets.md
    │   ├── model_comparison_results.csv
    │   └── cv_results.csv
    ├── scripts/
    │   ├── generate_data.py
    │   ├── analysis.py
    │   ├── ml_pipeline.py
    │   └── real_data_pipeline.py
    └── figures/
```

### [Layer A — Original Empirical Research](https://github.com/kofamaro450-a11y/Insecurity-in-Garissa-County-Analyses/tree/main/Layer-a-original-research)

*Assessment of Kenya's Current Insecurity Challenges: A Case Study of Security
Challenges in Garissa County.* A genuine mixed-methods study: 397 survey
responses, 20 key-informant interviews, 4 focus group discussions, documented
sampling methodology, reliability testing, qualitative thematic analysis, and an
established conceptual framework. This establishes that the underlying research
is real, original social science — not a dataset picked off Kaggle. **This layer
is untouched by the extension below** — the original research stands on its own.

### [Layer B — Computational Social Science Extension](layer-b-computational-extension/)

This project extends my undergraduate research using reproducible quantitative
and machine-learning workflows, in three parts:

1. A **reproducible rebuild of the thesis's real reported statistics**
   (descriptive figures + chi-square test), built from the actual numbers in
   Chapter Four.
2. A **synthetic-data ML pipeline demonstration** — preprocessing, feature
   engineering, a 75/25 train/test split, a majority-class baseline, three
   models (Logistic Regression, Decision Tree, Random Forest), 5-fold
   cross-validation, held-out evaluation, and interpretation. Necessary
   because respondent-level data from the original 2023 fieldwork was not
   retained — clearly labeled throughout as a methodological demonstration,
   not evidence about Garissa County.
3. A **real classification analysis on independently-sourced, government-
   published data**: Kenya's National Crime Research Centre publishes a
   public county-by-county crime-perception survey. Using 16 counties (10
   ASAL/frontier, including Garissa, plus 6 comparators) and a real,
   externally-defined target (frontier vs. non-frontier status), this part
   runs the same kind of pipeline — but with real data, a real target, and a
   real (if necessarily small-sample) result: Logistic Regression clearly
   beats baseline via Leave-One-Out cross-validation, and frontier counties
   show roughly double the murder-perception rate of non-frontier ones,
   consistent with the thesis's own qualitative themes.

Full write-ups and results tables for all three parts are in
[layer-b-computational-extension/README.md](layer-b-computational-extension/README.md).

## Why two layers

Graduate admissions committees can tell the difference between "I ran a model on
a dataset" and "I designed and executed original research, then built a
reproducible technical layer on top of it." Keeping these separate — with an
explicit, honest boundary between real findings, a synthetic-data
demonstration, and a real-but-small-sample analysis — is the more credible
story, not a weaker one.

## License

Code in `layer-b-computational-extension/scripts/` is MIT-licensed (see
`LICENSE`). The thesis text and figures in `layer-a-original-research/` are the
author's own academic work. External datasets referenced in
`layer-b-computational-extension/data/external_datasets.md`, and the NCRC data
underlying Part 3, are governed by their own respective sources' terms of use.
