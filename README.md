# Adversarial Robustness of ML/DL Intrusion Detection Systems

A reproducible **research codebase** for measuring how machine-learning and
deep-learning **network intrusion detection systems (IDS)** fail under
*constrained, feature-space evasion attacks*.

The pipeline trains detectors on **CIC-IDS2017-style flow features** (and
UNSW-NB15 / CSE-CIC-IDS2018 schemas), then attacks them with:

- a **conditional adversarial GAN** (budgeted, masked perturbations)
- **constrained FGSM and PGD** (same mask and `[0, 1]` box)
- a **transfer / surrogate** setup (craft on one model, score another)

and defends with **PGD adversarial training**.

This is a rebuild and research upgrade of
[tejaschaudhari131/adversarial-gan-ids](https://github.com/tejaschaudhari131/adversarial-gan-ids).
It is an **evaluation tool on already-extracted flow vectors**, not a live
packet exploit kit.

## Research question

> Under a documented grey-box / white-box / transfer threat model, how much
> does a flow-based IDS’s detection rate drop when an attacker may change
> only non-semantic features (timing, sizes, rates) inside a small L∞ budget?

Clean accuracy alone does not answer that. The primary numbers are
**evasion rate**, **attack success rate**, **accuracy drop**, and **mean L2**.
See [docs/METRICS.md](docs/METRICS.md) and [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

## Threat model (short)

| Cell | Attacker knowledge | Attacks in this repo |
| --- | --- | --- |
| White-box | Gradients of the target net | Constrained FGSM, PGD |
| Grey-box | Frozen differentiable IDS during training | Conditional GAN `G(x, z)` |
| Transfer | Surrogate only | PGD/GAN crafted on MLP → RF / CNN |

**Frozen:** destination port, TCP flags, protocol identifiers.
**Open:** duration, sizes, IATs, rates, window, active/idle stats.
**Budget:** `eps` in min-max scaled space; outputs clipped to `[0, 1]`.

A successful evasion here is *not* a guarantee that CICFlowMeter would emit
the perturbed vector from real packets (Pierazzi et al., IEEE S&P 2020).

## Datasets

| Name | Schema | First-class support |
| --- | --- | --- |
| CIC-IDS2017 | CICFlowMeter `MachineLearningCSV` (78 features) | Yes |
| CSE-CIC-IDS2018 | Abbreviated CICFlowMeter; recommended 3-day subset | Yes |
| UNSW-NB15 | Official train/test export | Yes |
| CIC-IoT-2023 | Stretch schema + synthetic stand-in | Loader only |

Exact download URLs, folder layout (`data/raw/...`), and the Engelen et al.
**CICIDS2017 labelling / CICFlowMeter caveats** are in
[docs/DATASETS.md](docs/DATASETS.md). Official dumps are **not** in git.

A **same-schema synthetic generator** is first-class so CI and a first clone
work without multi-GB downloads. Synthetic numbers are smoke-test artefacts,
not paper results.

Layout check and optional UNSW fetch (no CIC multi-GB downloads):

```bash
python run.py check-data --dataset-name unsw_nb15
python run.py setup-data --dataset-name unsw_nb15          # instructions only
python run.py setup-data --dataset-name unsw_nb15 --fetch  # public training CSV
python run.py prepare --dataset-name cicids2017            # fails with URLs if missing
```

Tiny committed schema fixtures (12 rows, not traffic) live under
`tests/fixtures/`. Use them to test loaders without official dumps.

## Method

1. Clean inf/NaN, drop duplicate `.1` / leakage columns, min-max scale,
   seeded train/val/test splits. Binary labels by default; multiclass
   encodings are stored alongside.
2. Fit a shared `IDSModel`: dropout **MLP**, **deeper MLP**, **1D CNN**,
   or sklearn **Random Forest**.
3. Attack only the attack-labelled rows, under the feature mask.
4. Optionally retrain the MLP with on-the-fly PGD (adversarial training).
5. Log hyperparameters + metrics to `results/` (JSON + plots).

## Quick start (no dataset download)

```bash
python -m pip install -r requirements.txt
python run.py pipeline --synthetic --quick
```

`--quick` shrinks the table and the epoch counts. A fuller synthetic run:

```bash
python run.py pipeline --synthetic
```

Artifacts go to `artifacts/` (scaler, IDS, GAN). Metrics and plots go to
`results/` (`evaluation_metrics.json`, `robustness_summary.png`).

Multi-model / multi-attack / transfer sweep (still synthetic, still minutes):

```bash
python run.py experiment-suite --config configs/quick.yaml
```

Longer multi-seed synthetic sweep (3 seeds, MLP + Random Forest, matched-`eps`
FGSM / PGD / GAN with L2 reported). Completes on CPU; writes real tables:

```bash
python run.py experiment-suite --config configs/medium.yaml
```

Real dumps (gitignored) after a successful fetch:

```bash
python scripts/download_datasets.py                 # UNSW official pair + Engelen CIC zip
python run.py experiment-suite --config configs/unsw_real.yaml
python run.py experiment-suite --config configs/cicids_engelen_friday.yaml
```

Official CIC `MachineLearningCSV.zip` hosts currently return an HTML portal;
the Engelen-corrected zip does not. Hashes and probe notes:
[docs/DATA_CARD.md](docs/DATA_CARD.md).

Aggregated CSV/Markdown from that run: [`results/tables/medium_synthetic_*.md`](results/tables/).
Do not treat those cells as CIC/UNSW paper results.

## Use real CSVs

```bash
# CIC-IDS2017 MachineLearningCSV (file or directory)
python run.py pipeline --dataset-name cicids2017 --dataset data/raw/cicids2017

# UNSW-NB15 official export
python run.py pipeline --dataset-name unsw_nb15 --dataset data/raw/unsw-nb15

# Constrained PGD against a trained MLP
python run.py train-ids --dataset data/raw/cicids2017 --ids-model mlp
python run.py evaluate --dataset data/raw/cicids2017 --attack pgd --eps 0.15
```

Stages: `prepare` · `train-ids` · `train-attack` / `train-gan` · `evaluate` ·
`pipeline` · `experiment-suite`.

Full-data sweep (falls back to synthetic if a path is missing):

```bash
python run.py experiment-suite --config configs/full.yaml
```

## Expected artefacts

| Path | What |
| --- | --- |
| `artifacts/dataset_meta.json` | Feature names, mask, split sizes, seed |
| `artifacts/ids_model.pt` / `*.joblib` | Fitted IDS |
| `artifacts/gan.pt` | Conditional generator |
| `results/evaluation_metrics.json` | Last pipeline metrics (real numbers from that run) |
| `results/<suite>/suite_metrics.md` | Attack × model × dataset table |
| `results/<suite>/config.json` | Hyperparameters actually used |

Empty paper tables live in `results/tables/` as **templates**. Fill them only
from runs you execute. The medium synthetic suite writes
`medium_synthetic_per_seed.*`, `medium_synthetic_aggregate.*`, and
`medium_synthetic_matched_eps.*` there.

## Project layout

```
run.py                      CLI
adv_ids/                    Research package (data, models, attacks, defenses)
configs/                    quick.yaml · medium.yaml · full.yaml · multi_dataset.yaml
docs/                       RESEARCH_PLAN · PAPER_OUTLINE · RELATED_WORK · THREAT_MODEL · METRICS · DATASETS
tests/                      preprocess, masks, metrics, attacks, fixtures, catalog
.github/workflows/ci.yml    ruff + pytest + synthetic pipeline
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tests and CI

```bash
python -m pip install -r requirements-dev.txt
ruff check adv_ids tests run.py
pytest -q
python run.py pipeline --synthetic --quick
```

GitHub Actions runs the same three steps on every push/PR.

## Documentation

| Doc | Contents |
| --- | --- |
| [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) | 8–12 week plan (literature → paper draft) |
| [docs/PAPER_OUTLINE.md](docs/PAPER_OUTLINE.md) | 2–3 month paper skeleton, figure/table plan, writing checklist |
| [docs/RELATED_WORK.md](docs/RELATED_WORK.md) | AdvGAN, IDSGAN, DEMGAN, constrained NIDS attacks, CICIDS caveats |
| [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) | White-box / grey-box / transfer, mask, non-claims |
| [docs/METRICS.md](docs/METRICS.md) | Evasion rate vs. ASR, L2, protocol |
| [docs/DATASETS.md](docs/DATASETS.md) | Download sources, folder layout, Engelen issues |
| [docs/DATA_CARD.md](docs/DATA_CARD.md) | Observed URLs, row counts, sha256 of files actually fetched |

## Resume-style summary

Built a reproducible Python/PyTorch research toolkit for adversarial robustness
of flow-based intrusion detection: CIC-IDS2017 / CSE-CIC-IDS2018 / UNSW-NB15
loaders with a synthetic CI stand-in; MLP, 1D-CNN, and Random Forest baselines;
constrained FGSM, PGD, and a conditional GAN under a protocol-feature mask;
PGD adversarial training and surrogate-to-target transfer; seeded YAML
experiment sweeps with JSON/plot artefacts. Framed as feature-space evaluation,
not a packet exploit.

## License

MIT. See [LICENSE](LICENSE).
