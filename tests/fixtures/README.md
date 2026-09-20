# Official-schema fixtures (tiny, committed)

These CSVs match published **column names**, not published traffic.
They exist so `prepare --dataset-name …` can be tested without multi-GB dumps.

| File | Mimics |
| --- | --- |
| `cicids2017_official_sample.csv` | CIC-IDS2017 `MachineLearningCSV` headers (leading spaces) |
| `cicids2018_official_sample.csv` | CSE-CIC-IDS2018 abbreviated CICFlowMeter names |
| `unsw_nb15_official_sample.csv` | Official `UNSW_NB15_training-set.csv` columns |

Regenerate with `python3 scripts/write_official_fixtures.py`.

Do not treat metrics on these 12-row files as experimental results.
