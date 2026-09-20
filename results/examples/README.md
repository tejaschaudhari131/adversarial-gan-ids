# Example metrics from runs that actually executed

These files were produced on this machine. They are **not** camera-ready
CIC-IDS2017 / UNSW-NB15 paper results unless the JSON says `synthetic: false`
**and** you repeat the run on the full official dump with `configs/full.yaml`.

```bash
python run.py pipeline --synthetic --quick
python run.py experiment-suite --config configs/quick.yaml
python run.py experiment-suite --config configs/medium.yaml
python run.py setup-data --dataset-name unsw_nb15 --fetch
# then a 2,300-row sample of the fetched official export (not committed)
```

| File | Source |
| --- | --- |
| `pipeline_synthetic_quick.json` | default GAN-vs-MLP pipeline (`eps=0.25`, 4 IDS + 8 GAN epochs) |
| `suite_synthetic_quick.md` | two-dataset smoke sweep |
| `suite_synthetic_medium.md` | 3-seed medium suite (66 rows); plots alongside |
| `medium_evasion_vs_l2.png` | matched-eps L2 vs evasion (synthetic) |
| `medium_aggregate_evasion.png` | mean ± std evasion bars (synthetic) |
| `medium_transfer_heatmap.png` | MLP→RF transfer (synthetic) |
| `pipeline_unsw_real_sample_quick.json` | earlier 2,300-row UNSW sample |
| `suite_unsw_real.md` | official 175,341 / 82,332 split, MLP, seed 42 |
| `unsw_real_attack_comparison.png` | FGSM / PGD / GAN on that official test set |
| `suite_cicids_engelen_friday.md` | Engelen Friday, 20k-row sample, MLP |
| `cicids_engelen_friday_attack_comparison.png` | FGSM / PGD / GAN on that sample |

Aggregated CSV/Markdown for the medium suite (committed, executed numbers only):
[`results/tables/medium_synthetic_*.md`](../tables/).

## Synthetic medium suite (seeds 42/43/44) — protocol check

Vanilla MLP, mean over 3 seeds. Source: `results/tables/medium_synthetic_aggregate.md`.

| dataset | attack | eps | evasion mean | L2 mean |
| --- | --- | --- | --- | --- |
| cicids2017 | fgsm | 0.15 | 0.9900 | 1.1975 |
| cicids2017 | pgd | 0.15 | 0.9900 | 1.1970 |
| cicids2017 | gan | 0.15 | 0.9850 | 1.1764 |
| cicids2017 | fgsm | 0.25 | 1.0000 | 1.9814 |
| cicids2017 | pgd | 0.25 | 1.0000 | 1.9125 |
| cicids2017 | gan | 0.25 | 1.0000 | 1.6646 |
| unsw_nb15 | fgsm | 0.15 | 0.9317 | 0.8609 |
| unsw_nb15 | pgd | 0.15 | 0.9317 | 0.8606 |
| unsw_nb15 | gan | 0.15 | 0.9133 | 0.8499 |

Adversarial training cut CIC stand-in PGD evasion at `eps=0.15` from 0.99 to
0.29 (clean acc 0.97). Transfer MLP→RF stayed weak on the CIC stand-in
(0.057 ± 0.023) and strong on the UNSW stand-in (0.715 ± 0.017).

## Official UNSW-NB15 train/test (full dumps, not committed)

`python run.py experiment-suite --config configs/unsw_real.yaml` after fetching
both official CSVs (see `docs/DATA_CARD.md`). Official split: train 157,806 +
val 17,535 from the 175,341-row training file; test = all 82,332 test rows;
42 features; scaler fit on train only. Seed 42, MLP 8 epochs, `eps=0.15`.

Source: `results/tables/unsw_real_per_seed.md`.

| attack | clean acc | clean F1* | evasion | drop | mean L2 |
| --- | --- | --- | --- | --- | --- |
| fgsm | 0.8437 | 0.8681 | 0.9851 | 0.5067 | 0.6285 |
| pgd | 0.8437 | 0.8681 | 0.9920 | 0.5103 | 0.5781 |
| gan (8 ep) | 0.8437 | 0.8681 | 0.1675 | 0.0725 | 0.4148 |

\*F1 from the suite log (`auc=0.9514`). One seed — not a multi-seed paper table.

## Engelen-corrected CIC-IDS2017 Friday (20k-row sample)

Full Friday file is 547,915 flows (gitignored). The suite used
`max_samples=20000` → train 14,000 / val 2,000 / test 4,000, 78 features after
aliasing Engelen names onto the CIC-IDS2017 list. Seed 42, MLP 6 epochs, `eps=0.15`.

Source: `results/tables/cicids_engelen_friday_per_seed.md`.

| attack | clean acc | evasion | drop | mean L2 |
| --- | --- | --- | --- | --- |
| fgsm | 0.9920 | 1.0000 | 0.4695 | 0.9605 |
| pgd | 0.9920 | 1.0000 | 0.4695 | 0.9019 |
| gan (6 ep) | 0.9920 | 1.0000 | 0.4695 | 0.4170 |

Friday-only + 20k cap. Do not cite as a full-week CIC-IDS2017 result.

Official `MachineLearningCSV.zip` URLs returned HTML portals in this VM
(`scripts/download_datasets.py --probe-cic-portal`).

## Earlier 2.3k UNSW sample (superseded by the official split above)

Fetched `UNSW_NB15_training-set.csv` from
`https://github.com/ushukkla/nospammers/raw/master/UNSW_NB15_training-set.csv`
(15,298,467 bytes, sha256 `7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c`,
82,332 rows = published testing-set size). Then sampled 1,500 benign + 800
attack rows to `/tmp` (not committed) and ran
`python run.py pipeline --dataset-name unsw_nb15 --dataset … --quick`.

| clean acc | clean F1 | evasion | drop | mean L2 | synthetic |
| --- | --- | --- | --- | --- | --- |
| 0.8674 | 0.8063 | 0.0787 | 0.0217 | 0.3340 | false |

The 8-epoch GAN barely moved this real-sample MLP. That is a training-budget
result on a 2.3k-row slice, not a claim about UNSW-NB15 robustness.

## Earlier synthetic smoke (still valid as a toolchain check)

`python run.py pipeline --synthetic --quick` (re-run after the medium suite):
clean acc 0.9978, evasion 0.9750, mean L2 1.4586.
