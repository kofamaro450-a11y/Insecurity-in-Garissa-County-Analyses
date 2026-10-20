# Relevant real-world datasets for follow-up / graduate-level work

**Update:** one of the datasets below (Kenya's National Crime Research Centre
county-level crime-perception survey) has now actually been used — see
`kenya_county_crime_perceptions_2020.csv` and Part 3 of the main notebook /
`scripts/real_data_pipeline.py`. It's kept in the list below too, since the
notes on its 2016/2018 waves and other counties remain useful for extending
that analysis further.

These are public datasets that align with this thesis's subject matter (insecurity,
terrorism, displacement, and socio-economic conditions in Garissa County / northeastern
Kenya). Most were not used to produce the figures in this repo — they're listed
here as a starting point for turning Appendix III's simulated demonstration into a real
empirical extension, or for a graduate-school research proposal that builds on this
thesis.

## 1. ACLED (Armed Conflict Location & Event Data Project) — Kenya conflict events
Event-level data on political violence, riots, protests, and fatalities, geocoded down
to locality, with actor tags (e.g. al-Shabaab, pastoralist militias, security forces).
This is the closest fit for quantifying the terrorism/banditry/inter-communal-conflict
trends this thesis discusses qualitatively — it would let a follow-up study build an
actual time series of Garissa County incidents (2015–present) instead of relying on
descriptive survey recall.
- Full dataset (registration required): https://acleddata.com/data-export-tool/
- Pre-packaged Kenya extract on the Humanitarian Data Exchange (no registration):
  https://data.humdata.org/dataset/kenya-acled-conflict-data
- ACLED's own Kenya situation reports (useful for triangulating and validating
  qualitative themes from Chapter Four): https://acleddata.com/region/africa/eastern-africa/kenya/

## 2. UNHCR Operational Data Portal / Refugee Data Finder — Dadaab refugee camp statistics
Population figures, demographics, and trends for the Dadaab refugee complex (in Garissa
County's Lagdera/Fafi sub-counties), directly relevant to the thesis's discussion of
refugee-related strain on local resources (§1.1, §3.3.1).
- Refugee Data Finder (global, filterable by country/camp):
  https://www.unhcr.org/refugee-statistics/download
- Kenya / Dadaab Operational Data Portal: https://data.unhcr.org/en/country/ken
- `refugees` R package (CRAN) wraps the same database programmatically:
  https://github.com/PopulationStatistics/refugees

## 3. Kenya National Bureau of Statistics (KNBS) — 2019 Census, Garissa County
Population, demographic, and socio-economic indicators at sub-county level for Garissa
— useful for weighting or contextualizing the quota-sampling design in §3.3.4–3.3.5,
and for a more rigorous sample-frame justification in a follow-up study.
- KNBS 2019 Kenya Population and Housing Census volumes: https://www.knbs.or.ke/
- County-level statistical abstracts (Garissa): search "Garissa County Statistical
  Abstract" on the KNBS site above.

## 4. Global Terrorism Database (GTD) — START, University of Maryland
Incident-level terrorism data since 1970 (perpetrator, target type, casualties,
weapon type), including the 2015 Garissa University College attack referenced
throughout Chapter One and Chapter Five. As of 2025 the GTD requires a data-request
form rather than a direct download.
- Access request: https://www.start.umd.edu/gtd/access/
- Codebook (useful for understanding variable definitions before requesting):
  https://www.start.umd.edu/gtd/downloads/Codebook.pdf

## 5. Afrobarometer — Kenya survey rounds
Nationally representative attitudinal survey data (including perceptions of security,
trust in police, and government performance) that could serve as an external benchmark
for this thesis's own survey findings on perceived government effectiveness (§4.3.2).
- https://www.afrobarometer.org/countries/kenya/

## 6. National Crime Research Centre (NCRC) — 47 County Data crime-perception survey [USED]
Kenya's National Crime Research Centre publishes a public, no-registration
county-by-county survey reporting the percentage of respondents who perceive
each of ~15-30 crime types as prevalent, for survey waves in 2016, 2018, and
2020. This is the dataset used in Part 3 of the main notebook — 16 counties
(10 ASAL/frontier including Garissa, plus 6 comparators), 2020 wave, manually
transcribed. All 47 counties have pages in the same format; extending to the
full set, or adding the 2016/2018 waves as a time dimension, is the natural
next step for strengthening that analysis.
- County pages: https://www.crimeresearch.go.ke/ (see "47 County Data" menu)
- Kenya-wide reports (Annual Crime Year Book, National Crime Mapping, etc.):
  https://www.crimeresearch.go.ke/reports/

## Suggested next step for a graduate research proposal

Pull ACLED event counts for Garissa County (and its five sub-counties) for 2015–2026,
aggregate to the same sub-county units used in this thesis's sampling frame (Table 3.1),
and test whether sub-county-level incident rates predict the "perceived security trend"
item that §A.4 of the thesis identifies as needing to be added to the questionnaire in
future fieldwork. That would replace the fully simulated logistic regression in
Appendix III with a real model linking objective conflict-event data to subjective
community perceptions — a natural extension for a master's thesis or research
assistantship application.
