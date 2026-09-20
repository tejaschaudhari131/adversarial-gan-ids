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

`python run.py prepare --dataset data/raw/cicids2017` concatenates every `*.csv`
under that directory. Column names are stripped; `Timestamp` / flow-id / IP
columns are dropped when present.

## CIC-IDS2017

- **Source:** [Canadian Institute for Cybersecurity — IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- **File to download:** `MachineLearningCSV.zip` (CICFlowMeter features, ~8 day CSVs)
- **Paper:** Sharafaldin, Lashkari, and Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” ICISSP 2018. [DOI 10.5220/0006639801080116](https://doi.org/10.5220/0006639801080116)
- **Schema:** 78 numeric flow features + `Label` (`BENIGN` vs. attack family)
- **CLI:** `python run.py pipeline --dataset-name cicids2017 --dataset data/raw/cicids2017`

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
- **CLI:** `python run.py pipeline --dataset-name unsw_nb15 --dataset data/raw/unsw-nb15 --synthetic`

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
