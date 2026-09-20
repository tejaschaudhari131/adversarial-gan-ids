# Paper outline (2–3 month student project)

Workshop / arXiv length (6–10 pages + appendix). Map every section to a file
in this repo so the write-up cannot drift from the code. Week numbers follow
[RESEARCH_PLAN.md](RESEARCH_PLAN.md).

This is an outline, not a draft. Do not paste synthetic smoke-test numbers
from `results/examples/` or `results/tables/medium_synthetic_*.md` into the
camera-ready main table.

## Title options

1. Constrained Feature-Space Evasion of Flow-Based Intrusion Detection Systems
2. How Much Does a Protocol-Feature Mask Protect ML-NIDS? A Multi-Dataset Study
3. Grey-Box GAN and White-Box PGD Attacks on CIC-IDS and UNSW-NB15 Detectors
4. Transfer Asymmetry in Adversarial Robustness of ML/DL Network IDS

Pick (1) or (2) unless the real-data transfer heatmap is the headline.

## Abstract skeleton (150–180 words)

- **Problem.** ML/DL flow-based IDS report high clean accuracy on CIC-IDS2017/2018
  and UNSW-NB15, but published numbers often ignore constrained adversaries and
  known CICFlowMeter / labelling errors.
- **Threat model.** White-box FGSM/PGD, grey-box conditional GAN, and
  surrogate-to-target transfer. Attacker may change timing/size/rate features
  inside an L∞ budget; ports, flags, and protocol IDs stay frozen.
  Feature-space success ≠ a valid packet (Pierazzi et al., IEEE S&P 2020).
- **Method.** Shared IDS interface (MLP, 1D-CNN, Random Forest); masked FGSM,
  PGD, and AdvGAN-style generator; PGD adversarial training; seeded configs.
- **Experiments.** Clean metrics + evasion rate + ASR + accuracy drop + mean L2
  on official (and, if used, Engelen-cleaned) dumps. Multi-seed. Matched-eps
  GAN vs PGD. Transfer heatmap.
- **Finding slot.** One sentence on the *real-data* ranking (not the synthetic
  smoke test). One sentence on the clean-accuracy cost of adversarial training.
- **Non-claim.** Not a live exploit kit.

## 1. Introduction  (week 10; freeze RQ in week 1)

- Why flow-feature IDS are the default academic benchmark.
- Clean accuracy is not robustness ([METRICS.md](METRICS.md)).
- Constrained domains: Sheatsley et al.; realistic NIDS power: Apruzzese et al.
  ([RELATED_WORK.md](RELATED_WORK.md), [THREAT_MODEL.md](THREAT_MODEL.md)).
- **Research question** (copy from README, do not silently rewrite later).
- Contributions: (i) unified masked-attack protocol; (ii) GAN vs PGD at
  matched L2; (iii) transfer cells reported separately; (iv) open config +
  `results/<run>/config.json` artefacts.
- Scope sentence: evaluation on extracted vectors, not packet mutation.

## 2. Related work  (week 1 notes → week 10 prose)

Use [RELATED_WORK.md](RELATED_WORK.md) as the bibliography source. Subsections:

- Datasets and hygiene (Sharafaldin 2018; Engelen 2021; Moustafa 2015).
- Gradient attacks and Madry / TRADES training.
- AdvGAN, IDSGAN, NIDSGAN, DEMGAN — what we reuse vs. what we do not claim.
- Constrained / problem-space attacks (Sheatsley, Pierazzi, Hashemi).

End with a table: “this paper vs. IDSGAN / DEMGAN / Madry-NIDS” on threat
model, mask, datasets, and open code.

## 3. Threat model  (week 1; do not edit after week 2)

Almost a lift of [THREAT_MODEL.md](THREAT_MODEL.md):

- Goal = test-time evasion of *already detected* attacks.
- Knowledge cells: white-box / grey-box / transfer.
- Capability: feature mask, `[0,1]` box, `eps`.
- Defender: labelled flows; optional PGD adv-training.
- Non-claims: no NIC-level realisation.

## 4. Method  (weeks 3–7; cite `adv_ids/`)

| Paper subsection | Code |
| --- | --- |
| 4.1 Datasets and cleaning | `adv_ids/data/preprocess.py`, [DATASETS.md](DATASETS.md) |
| 4.2 IDS models | `adv_ids/models/` (`mlp`, `deep_mlp`, `cnn1d`, `random_forest`) |
| 4.3 Constraints | `adv_ids/attacks/constraints.py`, `adv_ids/data/masks.py` |
| 4.4 FGSM / PGD | `adv_ids/attacks/fgsm.py`, `pgd.py` |
| 4.5 Conditional GAN | `adv_ids/attacks/gan_attack.py` |
| 4.6 Adversarial training | `adv_ids/defenses/adv_train.py` |
| 4.7 Transfer | `adv_ids/attacks/transfer.py`, suite `transfer:` rows |
| 4.8 Metrics | `adv_ids/evaluation/metrics.py`, [METRICS.md](METRICS.md) |

Equations: masked L∞ PGD; GAN loss = fool-D + λ_ids P(attack) + λ_pert ‖δ⊙m‖².

## 5. Experimental protocol  (weeks 2, 8, 11)

- Hardware, seeds (`configs/full.yaml` `seeds:`), commit hash.
- Data card: URLs, dates, `sha256sum`, row counts after inf/NaN drop
  (`python run.py check-data --dataset-name …`).
- Official vs. Engelen-cleaned CIC-IDS2017 — say which, never both unlabeled.
- Train / val / test ratios; binary vs. multiclass (attacks are binary).
- Matched-`eps` rule: always report mean L2 next to evasion
  (`results/tables/*_matched_eps.md`).
- Synthetic stand-in is **appendix / CI only**.

Commands:

```bash
python run.py experiment-suite --config configs/medium.yaml   # synthetic protocol check
python run.py experiment-suite --config configs/full.yaml     # real CSVs
```

## 6. Results  (weeks 4–9, 11)

Fill from executed `results/<run>/` only.

**Table 1.** Clean accuracy / F1 / AUC by dataset × model (mean ± std, 3 seeds).

**Table 2.** Matched-eps attack comparison: FGSM, PGD, GAN — evasion, ASR,
accuracy drop, mean L2. One block per dataset. Source pattern:
`results/tables/<run>_matched_eps.md`.

**Table 3.** Vanilla MLP vs. adversarially trained MLP.

**Table 4.** Transfer heatmap values (surrogate → target), not averaged into
Table 2.

**Figure 1.** Clean confusion + ROC (`confusion_matrix_clean.png`).

**Figure 2.** Evasion vs. mean L2 scatter (`evasion_vs_l2.png`) — the figure
that justifies “GAN vs PGD at matched cost.”

**Figure 3.** Multi-seed evasion bars with error bars (`aggregate_evasion.png`).

**Figure 4.** Transfer heatmap (`transfer_heatmap.png`).

Reserve a short **negative-results** paragraph (GAN under-trained, mask
ablation that did not help, etc.).

## 7. Discussion

- Feature-space vs. problem-space (what a 0.15 shift in `Flow Bytes/s` means).
- Why transfer can be asymmetric (document, do not over-theorise).
- CICIDS hygiene: would Engelen labels change the ranking?
- Limitations: binary head, no certified defense, no packet replay.

## 8. Conclusion

Restate the RQ, the strongest *real-data* cell, and the open artefacts
(`configs/`, `docs/`, `results/<run>/config.json`).

## Appendix

- A: hyper-parameters (dump of `config.json`).
- B: full per-seed CSV (`*_per_seed.csv`).
- C: synthetic protocol check (point at `results/tables/medium_synthetic_*.md`,
  labelled as such).
- D: dataset layout and checksums.

## Week-by-week writing checklist

Tied to [RESEARCH_PLAN.md](RESEARCH_PLAN.md):

| Week | Writing task | Done when |
| --- | --- | --- |
| 1 | Freeze title + RQ + threat-model paragraph | pasted into §1 and §3, not edited later |
| 2 | Data card (URLs, hashes, counts) | Appendix D exists |
| 3 | Clean-baseline table caption | Table 1 skeleton with *real* numbers |
| 4 | Attack-comparison figure caption | Figure 2 from a real `eps` sweep |
| 5 | GAN vs PGD paragraph | Table 2 includes both at matched L2 |
| 6 | Transfer subsection | Table 4 + “do not average” sentence |
| 7 | Defense trade-off paragraph | Table 3 |
| 8 | Multi-dataset confirmation | Table 2 has ≥2 datasets |
| 9 | Negative-results subsection | at least one failed ablation described |
| 10 | Related work + intro + method | citations only from RELATED_WORK.md |
| 11 | Second/third seed, stdev in tables | `n_seeds ≥ 3` in aggregate tables |
| 12 | Camera-ready: no synthetic main-table cells; no copied DEMGAN % | peer can rerun `configs/full.yaml` |

## What this outline must not become

- A live-packet “bypass the SOC” paper.
- A leaderboard of synthetic `--quick` metrics.
- A related-work dump of unread PDFs.
