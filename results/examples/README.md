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
| `pipeline_unsw_real_sample_quick.json` | GAN-vs-MLP on a 2,300-row sample of the fetched official UNSW export |

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

## Real UNSW official-export sample (not the full dump)

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
