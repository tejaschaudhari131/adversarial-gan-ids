# Research plan (8–12 weeks)

A student-sized plan that this repository is meant to support. Weeks are
sequential; overlap literature with engineering in week 1–2. The code already
covers the scaffolding for weeks 3–7 so calendar time can go to *real data*,
*seeds*, and *writing*, not to rewriting loaders.

## Week 1 — Problem, threat model, literature

- Read [THREAT_MODEL.md](THREAT_MODEL.md) and [RELATED_WORK.md](RELATED_WORK.md).
- Pull the primary PDFs: Sharafaldin 2018, Engelen 2021, Lin/IDSGAN, Xiao/AdvGAN,
  Madry 2018, Sheatsley 2022, Apruzzese 2022, Hou/DEMGAN 2025.
- Write a one-page note: research question, non-goals (no live exploit),
  success metric (evasion rate + L2, not accuracy alone).
- Reproduce the synthetic smoke test; keep that log as your “toolchain works”
  artefact.

**Exit:** annotated bibliography (20–30 entries) and a frozen threat-model
paragraph you will not silently change later.

## Week 2 — Data, hygiene, splits

- Download CIC-IDS2017 `MachineLearningCSV` and the Engelen reconstructed
  features if the host still serves them.
- Download a **subset** of CSE-CIC-IDS2018 (the three days listed in
  [DATASETS.md](DATASETS.md)) and the UNSW-NB15 official train/test CSVs.
- Do **not** commit the dumps. Put them under `data/raw/...`.
- Run `python run.py prepare --dataset-name cicids2017 --dataset data/raw/cicids2017`
  and inspect `artifacts/dataset_meta.json` (class balance, dropped inf/NaN).
- Decide: official CIC dump vs. Engelen-cleaned (prefer cleaned for the paper).
- Fix seeds, train/val/test ratios, and whether rare classes are dropped or
  grouped (“Other”).

**Exit:** a `docs/DATA_CARD.md` you write yourself: exact URLs, dates, file
hashes, row counts after cleaning, label map.

## Week 3 — Clean baselines

- Train `mlp`, `deep_mlp`, `cnn1d`, `random_forest` on each dataset
  (`configs/full.yaml` is the starting sweep).
- Log clean accuracy / precision / recall / F1 / AUC. Compare to published
  CICIDS baselines — large gaps usually mean a split or leakage bug, not a
  new architecture.
- Optional: add XGBoost/LightGBM if the RF vs. MLP gap needs a stronger
  tree baseline. Keep the shared `IDSModel` interface.

**Exit:** a clean-performance table with seeds. No adversarial numbers yet.

## Week 4 — Constrained white-box attacks

- Sweep `eps ∈ {0.05, 0.10, 0.15, 0.25}` for FGSM and PGD (10 / 20 / 40 steps).
- Confirm frozen features do not move (`max_frozen_feature_change ≈ 0`).
- Plot evasion rate vs. mean L2 (the interesting curve).
- Ablate the mask: “ports+flags frozen” vs. “only ports frozen” vs. none.
  The last setting is an *upper bound*, not a realistic attacker.

**Exit:** attack-comparison figure on CIC-IDS2017 (or the cleaned dump).

## Week 5 — GAN attack and training stability

- Train the conditional GAN against each frozen differentiable IDS.
- Watch `IDS_P(attack)` and mean L2; if the generator collapses to the
  maximum budget, raise `lambda_pert` or lower `eps`.
- Compare GAN vs. PGD at matched L2, not matched epoch count.
- Optional stretch: WGAN-GP critic or a DEMGAN-style distortion term. Only
  add this if PGD is already solid.

**Exit:** GAN vs. PGD table + a short “when the GAN helps” note.

## Week 6 — Transfer / grey-box

- Craft PGD (and GAN) on MLP, evaluate on CNN, deep MLP, and Random Forest.
- Reverse the arrows (RF has no gradients — use the MLP as surrogate only).
- Document **asymmetry**. Do not average transfer cells into the white-box
  mean.
- Optional: query-only substitute (train an MLP on target hard labels).

**Exit:** transfer heatmap and a paragraph on what a remote attacker could
actually assume.

## Week 7 — Defenses

- PGD adversarial training on the MLP (`defense.name: adv_train`).
- Measure clean F1 cost vs. evasion-rate gain (the Madry trade-off).
- Optional: TRADES (`β` sweep) if you have GPU time.
- Sanity: adv-training must not be evaluated only on the same `eps` it saw;
  include a held-out larger `eps`.

**Exit:** defense table (clean / FGSM / PGD / GAN) for vanilla vs. robust MLP.

## Week 8 — Multi-dataset confirmation

- Repeat the *winning* attack/defense pair on CSE-CIC-IDS2018 (subset) and
  UNSW-NB15. Do not claim “general robustness” from CIC-IDS2017 alone.
- If feature spaces differ, **do not** force a joint model; report per-dataset
  tables. Cross-dataset transfer is only meaningful after feature alignment
  (CIC 2017/2018).
- Optional stretch: CIC-IoT-2023 day sample.

**Exit:** three-dataset robustness table.

## Week 9 — Ablations and negative results

- Budget, mask, λ_ids, generator depth, IDS capacity, class imbalance.
- Write down failures (GAN worse than PGD; RF more robust to transfer; etc.).
  Negative results are expected and publishable if the protocol is clean.

**Exit:** ablation appendix + a “what did *not* work” subsection.

## Week 10 — Writing

- Related work from [RELATED_WORK.md](RELATED_WORK.md); threat model from
  [THREAT_MODEL.md](THREAT_MODEL.md); metrics from [METRICS.md](METRICS.md).
- Figures: robustness bars, evasion-vs-L2, transfer heatmap, confusion
  matrices. All generated from `results/<run>/`, not redrawn by hand.
- Repro appendix: commit hash, configs, hardware, seeds.

**Exit:** draft (workshop / arXiv length).

## Weeks 11–12 — Hardening the claim

- Second seed (or three).
- Peer check: can a classmate rerun `configs/full.yaml` from the data card?
- Optional: problem-space discussion (what it would take to realise a
  perturbation in packets) — analysis only, still not an exploit kit.
- Camera-ready checklist: no synthetic numbers in the main table; no
  unlabelled DEMGAN percentages copied from their PDF.

## Out of scope for this plan

- Live packet mutation, C2 tooling, or “bypassing a production IDS”.
- Training foundation models on raw pcap.
- Claiming SOTA on CIC-IDS2017 without the Engelen caveat.
