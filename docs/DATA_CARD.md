# Data card (observed in this environment, 2026-09-20)

Filled from downloads that actually completed. Do not commit the dumps.

## UNSW-NB15 official ML export

| File | Rows | Bytes | sha256 | Source |
| --- | --- | --- | --- | --- |
| `data/raw/unsw-nb15/UNSW_NB15_training-set.csv` | 175,341 | 32,293,018 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` | https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv |
| `data/raw/unsw-nb15/UNSW_NB15_testing-set.csv` | 82,332 | 15,298,467 | `7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c` | https://github.com/ushukkla/nospammers/raw/master/UNSW_NB15_training-set.csv (filename says training; **row count is the published test set**) |

Official homepage: https://research.unsw.edu.au/projects/unsw-nb15-dataset  
Nir-Az raw GitHub training URL: HTTP 404.

`prepare --dataset-name unsw_nb15 --dataset data/raw/unsw-nb15` uses this pair as the **official split** (scaler fit on train only).

## CIC-IDS2017

### Official MachineLearningCSV.zip — blocked here

Probed 2026-09-20:

- `http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip`
- `https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip`

Both returned **HTTP 200 + `text/html` (~108,784 bytes)** — a CIC/UNB landing page with GTM, not a zip. Treat as registration/portal, not a silent fetch success.

Homepage: https://www.unb.ca/cic/datasets/ids-2017.html

### Engelen-corrected regeneration — fetched here

| File | Bytes | sha256 | Source |
| --- | --- | --- | --- |
| `data/raw/cicids2017/engelen_WTMC2021_dataset.zip` | 333,841,436 | `4d535da19795d85376ae1397d161329e3b06fc47d9a5a68cd9be2cd7ecee0f2a` | https://intrusion-detection.distrinet-research.be/WTMC2021/Dataset/dataset.zip |

Contents: `Monday|Tuesday|Wednesday|Thursday|Friday-WorkingHours.csv` (~1.15 GB uncompressed).

This VM extracted **all five days**. Friday also sits at the zip top level.

| Day file | Rows | Notes |
| --- | --- | --- |
| `Friday-WorkingHours.csv` (top-level + `week/`) | 547,915 | BENIGN 291433, PortScan 159151, DDoS 95123, Bot 738 + Attempted |
| `week/Monday-WorkingHours.csv` | 371,749 | extracted locally from the same zip |
| `week/Tuesday-WorkingHours.csv` | 322,003 | |
| `week/Wednesday-WorkingHours.csv` | 496,779 | |
| `week/Thursday-WorkingHours.csv` | 362,368 | |
| **Week concat** | **2,100,814** | `data/raw/cicids2017/week/` |

Long suites: `configs/cicids_engelen_long.yaml` (Friday + week) and
`configs/cicids_engelen_friday_long.yaml` (Friday only). Dumps stay gitignored.

Engelen headers use fixed-CICFlowMeter names (`Dst Port`, `Total Fwd Packet`, `FWD Init Win Bytes`); the loader aliases them onto the CIC-IDS2017 feature list.

## CSE-CIC-IDS2018

Not downloaded (AWS Open Data, multi-GB day files). Script: place 2–3 days under `data/raw/cse-cic-ids2018/`.

## Commands

```bash
python scripts/download_datasets.py
python scripts/download_datasets.py --probe-cic-portal
python run.py setup-data --dataset-name unsw_nb15 --fetch
python run.py setup-data --dataset-name cicids2017 --fetch
```
