# Raw datasets (not committed)

Place official CSVs here. See [docs/DATASETS.md](../../docs/DATASETS.md) and
`adv_ids/data/catalog.py` for homepages, example filenames, and checksum notes.

```
data/raw/
  cicids2017/                 # MachineLearningCSV day files (hundreds of MB)
  cse-cic-ids2018/            # AWS Open Data day CSVs (GB each — 2–3 days only)
  unsw-nb15/                  # UNSW_NB15_training-set.csv / testing-set.csv
  cic-iot-2023/               # optional stretch
```

```bash
python run.py check-data --dataset-name cicids2017
python run.py setup-data --dataset-name unsw_nb15 --fetch
```

`--fetch` is only wired for UNSW (tens of MB). CIC dumps are not auto-downloaded.
Without these folders, use `--synthetic` or the tiny fixtures in
`tests/fixtures/` (schema only, not traffic).

Do not commit zips, pcaps, or official CSVs. After a download, record
`sha256sum` in a local `docs/DATA_CARD.md`.
