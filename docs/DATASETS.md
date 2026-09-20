# Datasets and on-disk layout

This project evaluates **already-extracted flow feature vectors**. Do not commit
the multi-GB official dumps. Scripts accept a file or a directory of CSVs and
will generate a same-schema synthetic stand-in when you pass `--synthetic`.

## Folder layout

```
data/
  raw/
    cicids2017/                 # CIC-IDS2017 MachineLearningCSV day files
      Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
      ...
    cse-cic-ids2018/            # CSE-CIC-IDS2018 AWS CSV subset (a few days)
      Wednesday-14-02-2018.csv
      Thursday-15-02-2018.csv
      Friday-16-02-2018.csv
    unsw-nb15/                  # Official ML export
      UNSW_NB15_training-set.csv
      UNSW_NB15_testing-set.csv
    cic-iot-2023/               # Optional stretch
      *.csv
  synthetic_cicids2017.csv      # generated locally; gitignored
  processed/                    # optional cached tables; gitignored
```

`python run.py prepare --dataset-name cicids2017 --dataset data/raw/cicids2017`
concatenates every `*.csv` under that directory. Column names are stripped;
`Timestamp` / flow-id / IP columns are dropped when present.

If the folder is empty or missing, `prepare` / `check-data` / `setup-data`
exit with code 2 and print homepage, example filenames, checksum notes, and
the next command. Do not guess a path.

```bash
python run.py check-data --dataset-name cicids2017
python run.py check-data --dataset-name cicids2018
python run.py check-data --dataset-name unsw_nb15
python run.py setup-data --dataset-name unsw_nb15 --fetch   # optional public training CSV
```

Catalog + layout live in `adv_ids/data/catalog.py`. Tiny official-schema
fixtures (12 rows, committed) are under `tests/fixtures/` so loaders can be
tested without multi-GB dumps:

```bash
python run.py prepare --dataset-name unsw_nb15 \
  --dataset tests/fixtures/unsw_nb15_official_sample.csv
```

## CIC-IDS2017

- **Source:** [Canadian Institute for Cybersecurity — IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- **File to download:** `MachineLearningCSV.zip` (CICFlowMeter features, ~8 day CSVs)
- **Paper:** Sharafaldin, Lashkari, and Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” ICISSP 2018. [DOI 10.5220/0006639801080116](https://doi.org/10.5220/0006639801080116)
- **Schema:** 78 numeric flow features + `Label` (`BENIGN` vs. attack family)
- **CLI:** `python run.py pipeline --dataset-name cicids2017 --dataset data/raw/cicids2017`
- **Official zip in this VM:** `MachineLearningCSV.zip` URLs returned HTML
  portals (~109 KB, `text/html`), not the archive. See [DATA_CARD.md](DATA_CARD.md).
- **Engelen-corrected zip (no login, verified):**
  `https://intrusion-detection.distrinet-research.be/WTMC2021/Dataset/dataset.zip`
  `sha256=4d535da19795d85376ae1397d161329e3b06fc47d9a5a68cd9be2cd7ecee0f2a`
  (333,841,436 bytes).  
  `python run.py setup-data --dataset-name cicids2017 --fetch` downloads it and
  extracts day CSVs. Do not commit the zip or the CSVs.
  Small real eval: `python run.py experiment-suite --config configs/cicids_engelen_friday.yaml`

### Known issues (do not ignore these in a paper)

Engelen, Rimmer, and Joosen, “Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study,” IEEE SPW 2021 ([DOI 10.1109/SPW53761.2021.00009](https://doi.org/10.1109/SPW53761.2021.00009); [project page](https://intrusion-detection.distrinet-research.be/WTMC2021/)) documented errors in:

- CICFlowMeter TCP termination / flow construction
- Feature extraction (including duplicate `Fwd Header Length` / `Fwd Header Length.1`)
- Labelling of attempted vs. successful attacks
- Infinite / undefined `Flow Bytes/s` and `Flow Packets/s`

They reconstructed or relabelled more than 20% of the original traces. A corrected
flow-feature release is linked from that page. **Prefer the corrected pipeline
when you write the paper.** This repo still accepts the official
`MachineLearningCSV` files and applies practical cleaning (strip headers, drop
`.1` duplicates, drop inf/NaN, min-max scale). That is *not* a substitute for
the Engelen et al. regeneration.

## CSE-CIC-IDS2018

- **Source:** [CIC — IDS 2018](https://www.unb.ca/cic/datasets/ids-2018.html)
- **AWS Open Data:** [registry.opendata.aws/cse-cic-ids2018](https://registry.opendata.aws/cse-cic-ids2018/)
- **Recommended subset (tractable, diverse):**
  - `Wednesday-14-02-2018` — FTP-BruteForce, SSH-Bruteforce
  - `Thursday-15-02-2018` — DoS-GoldenEye, DoS-Slowloris
  - `Friday-16-02-2018` — DoS-SlowHTTPTest, DoS-Hulk
- **Schema:** abbreviated CICFlowMeter names (`Dst Port`, `Tot Fwd Pkts`, …).
  The loader maps these onto the CIC-IDS2017 names when both are present.
- Engelen et al. note that several CICFlowMeter issues also affect 2018.

## UNSW-NB15

- **Source:** [UNSW Canberra — UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Paper:** Moustafa and Slay, “UNSW-NB15: a comprehensive data set for network intrusion detection systems,” MilCIS 2015. [DOI 10.1109/MilCIS.2015.7348942](https://doi.org/10.1109/MilCIS.2015.7348942)
- **Preferred files:** `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`
  (42 features + `attack_cat` + binary `label`). This repo uses the numeric
  columns and factorizes `proto` / `service` / `state`.
- **CLI:** `python run.py pipeline --dataset-name unsw_nb15 --dataset data/raw/unsw-nb15`
- **Optional fetch:** `python run.py setup-data --dataset-name unsw_nb15 --fetch`
  or `python scripts/download_datasets.py`. Observed hashes live in
  [DATA_CARD.md](DATA_CARD.md).
- **Working mirrors (verified 2026-09-20):**
  - Training (175,341 rows, 32.3 MB):
    `https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv`
    `sha256=bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`
  - Testing (82,332 rows, 15.3 MB; file may be *named* training-set):
    `https://github.com/ushukkla/nospammers/raw/master/UNSW_NB15_training-set.csv`
    `sha256=7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c`
  - Nir-Az raw GitHub URL: HTTP 404 here.
- When both official files sit in `data/raw/unsw-nb15/`, `prepare` uses that
  **official split** (scaler + categorical codes fit on train only).
- **Fixture:** `tests/fixtures/unsw_nb15_official_sample.csv` matches the
  official column order (`id` … `attack_cat`,`label`).

## CIC-IoT-2023 (optional stretch)

- **Source:** [CIC — IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- **Paper:** Neto et al., “CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment,” *Sensors* 23(13):5941, 2023. [DOI 10.3390/s23135941](https://doi.org/10.3390/s23135941)
- A synthetic stand-in with a compact numeric schema is wired so the loader
  path can be tested without the download. Full-scale IoT sweeps are left for
  the 2–3 month research window.

## Synthetic stand-in

```bash
python run.py prepare --synthetic --dataset-name cicids2017
python run.py prepare --synthetic --dataset-name unsw_nb15
```

The generator matches the **column schema** and injects a separable shift on
modifiable features so models and attacks have a signal. It is **not** a
substitute for published traffic and must not be reported as CIC/UNSW results.

## Label modes

- `binary` (default): benign / Normal / 0 → `0`, everything else → `1`
- `multiclass`: encodes the raw attack family (`Label` or `attack_cat`). Clean
  metrics work; evasion attacks still target the benign class conceptually.
  Gradient attacks in this repo are implemented for the binary head.
