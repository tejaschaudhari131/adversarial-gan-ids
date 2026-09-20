# Constrained Feature-Space Evasion of Flow-Based Intrusion Detection Systems

**Working draft** for a 2–3 month student project. Workshop / arXiv length.
Every numeric cell below is copied from an executed run in this repository
(`results/tables/`). Empty or missing cells were not measured. Synthetic
smoke-test numbers are **not** used in the main tables.

Outline map: [PAPER_OUTLINE.md](../PAPER_OUTLINE.md).
Code: `adv_ids/`. Configs: `configs/unsw_real_long.yaml`,
`configs/cicids_engelen_long.yaml`.

---

## Abstract

Machine-learning intrusion detection systems (IDS) trained on flow features
from CIC-IDS2017 and UNSW-NB15 routinely report high clean accuracy, but
those numbers say little about a constrained adversary who may change only
timing, size, and rate features inside a small L∞ budget. We evaluate
dropout MLPs and Random Forests under a documented threat model: white-box
FGSM/PGD, a grey-box AdvGAN-style conditional generator, and surrogate
transfer onto a non-differentiable forest. Protocol identifiers, ports, and
TCP flags stay frozen; outputs are clipped to the min-max cube.

On the **official UNSW-NB15 train/test export** (175,341 / 82,332 rows;
three seeds), a 10-epoch MLP reaches mean clean accuracy 0.864. Constrained
PGD at `eps=0.15` evades 0.931 ± 0.060 of previously detected attacks
(mean L2 0.552). A 24-epoch GAN at the same `eps` evades only 0.377 ± 0.133
at a smaller L2 (0.328). Matching on L2 rather than `eps` — GAN `eps=0.15`
versus PGD `eps=0.10`, L2 gap 0.050 — still leaves PGD ahead (0.875 vs
0.377 evasion). Transfer of PGD from the MLP onto Random Forest is much
weaker (0.156 ± 0.085).

CIC-IDS2017 is evaluated on the **Engelen-corrected** regeneration, not the
official MachineLearningCSV zip (those hosts returned an HTML portal here).
A prior 20k-row Friday smoke test is on disk; a full-Friday plus five-day
multi-seed sweep is the companion experiment (`cicids_engelen_long`).
Feature-space success is not a packet. This is not a live exploit kit.

---

## 1. Introduction

Flow-feature IDS benchmarks (CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15) are
the default academic stand-in for “network intrusion detection.” Clean
accuracy on those tables is often in the high 90s. That is the wrong
primary number for robustness [METRICS.md](../METRICS.md).

An attacker who can rewrite every coordinate of a CICFlowMeter vector is
unrealistic. Ports, flags, and protocol IDs are semantically sticky;
duration, sizes, IATs, and rates are not. Sheatsley et al. (JCS 2022) show
that a tiny allowed feature set can still evade. Pierazzi et al. (IEEE S&P
2020) warn that a successful feature-space evasion need not correspond to
any valid packet. Apruzzese et al. (DTRAP 2022) argue that many NIDS papers
grant the attacker oracles a remote adversary does not have.

**Research question.** Under a documented white-box / grey-box / transfer
threat model, how much does a flow-based IDS’s detection rate drop when the
attacker may change only non-semantic features inside a small L∞ budget?

**Contributions.**

1. A unified masked-attack protocol (FGSM, PGD, conditional GAN) on a
   shared `IDSModel` interface (`adv_ids/`).
2. GAN vs PGD compared at **closest matched mean L2**, not only matched
   `eps` (`results/tables/*_matched_l2.md`).
3. Transfer cells reported separately and never averaged into the
   white-box mean.
4. Open configs and `results/<run>/config.json` artefacts; official UNSW
   split; Engelen CIC-IDS2017 aliases.

**Scope.** Evaluation on already-extracted vectors. No NIC-level
realisation, no Suricata/Snort bypass.

---

## 2. Related work

Citations follow [RELATED_WORK.md](../RELATED_WORK.md).

### 2.1 Datasets and hygiene

Sharafaldin, Lashkari, and Ghorbani (ICISSP 2018) released CIC-IDS2017 as
CICFlowMeter day CSVs. Moustafa and Slay (MilCIS 2015) released UNSW-NB15
with an official 175,341 / 82,332 train/test export. Engelen, Rimmer, and
Joosen (IEEE SPW 2021) showed that CICFlowMeter termination, duplicate
`Fwd Header Length`, infinite rates, and attempted-vs-successful labels
corrupt CIC-IDS2017 rankings and published a reconstructed dump. This
draft uses that Engelen regeneration for CIC and the official ML export
for UNSW. Official CIC `MachineLearningCSV.zip` URLs returned HTML
portals in this environment ([DATA_CARD.md](../DATA_CARD.md)).

### 2.2 Gradient attacks and training

FGSM (Goodfellow et al., ICLR 2015) and PGD / adversarial training
(Madry et al., ICLR 2018) are the white-box baselines. We add a feature
mask and a `[0,1]` box; they are constrained feature-space attacks, not
ImageNet copies. TRADES (Zhang et al., ICML 2019) is left as future
defense work.

### 2.3 GAN evasion

AdvGAN (Xiao et al., IJCAI 2018) trains a generator against a frozen
classifier. IDSGAN (Lin, Shi, Xue, PAKDD 2022) applies a WGAN plus
restricted functional features on NSL-KDD. DEMGAN (Hou et al., 2025)
reports high evasion on CIC-IDS2017/2018; those percentages are *theirs*,
not reproduced here. NIDSGAN (Shu et al., 2022) adds domain constraints
and limited oracle access. Our generator is the AdvGAN recipe on flow
vectors (fool-D + λ_ids P(attack) + L2 penalty + mask), not a WGAN-GP
or multi-generator DEMGAN.

### 2.4 Constrained / problem-space work

Sheatsley et al. (arXiv:2011.01183; JCS 2022): constrained-domain
adaptive attacks on NSL-KDD / UNSW-NB15. Pierazzi et al. (S&P 2020):
feature-space ≠ problem-space. Hashemi, Cusack, Keller (Big-DAMA 2019):
NIDS in an adversarial setting. Apruzzese et al. (DTRAP 2022): attacker
power axes.

| Line of work | Reused here | Not claimed |
| --- | --- | --- |
| AdvGAN / IDSGAN / DEMGAN | Conditional G, frozen IDS, feature mask | WGAN-GP, DEMGAN %, NSL-KDD |
| FGSM / PGD / Madry | Masked first-order attacks | TRADES / certificates |
| Sheatsley / Pierazzi | Mask + box + honesty clause | JSMA, packet mutation |
| Apruzzese | White-box / grey-box / transfer cells | “PGD is operationally realistic” |

---

## 3. Threat model

Lifted from [THREAT_MODEL.md](../THREAT_MODEL.md). Frozen after this
section.

**Goal.** Test-time evasion of *already detected* attack flows.

| Cell | Knowledge | Implementation |
| --- | --- | --- |
| White-box | Weights and gradients of the target net | Constrained FGSM, PGD |
| Grey-box | Frozen differentiable IDS during G training | Conditional GAN `G(x,z)` |
| Transfer | Surrogate only | PGD crafted on MLP → Random Forest |

**Capability.** (i) Feature mask: destination port, TCP flags, protocol
IDs frozen (`adv_ids/data/schemas.py`). (ii) Box: scaled `[0,1]`.
(iii) Budget: L∞ `eps`. (iv) Only attack rows are perturbed.

**Defender.** Labelled flows; optional PGD adversarial training (not the
headline of this draft’s long runs).

**Non-claim.** A successful cell is not a valid packet and is not a
bypass of a deployed NIDS stack. White-box PGD is an **upper bound** on
feature-space vulnerability.

---

## 4. Method

| Subsection | Code |
| --- | --- |
| 4.1 Data | `adv_ids/data/preprocess.py`, `catalog.py`, `setup.py` |
| 4.2 Models | `adv_ids/models/` (`mlp`, `random_forest`, …) |
| 4.3 Constraints | `adv_ids/attacks/constraints.py`, `data/masks.py` |
| 4.4 FGSM / PGD | `adv_ids/attacks/fgsm.py`, `pgd.py` |
| 4.5 GAN | `adv_ids/attacks/gan_attack.py` |
| 4.6 Transfer | `adv_ids/attacks/transfer.py` |
| 4.7 Metrics | `adv_ids/evaluation/metrics.py` |

**Cleaning.** Inf/NaN drop; leakage columns (IPs, flow id, timestamp,
source port) dropped; Engelen names aliased onto the CIC-IDS2017 list;
UNSW `proto`/`service`/`state` factorized on **train only** for the
official split; min-max scale fitted on train.

**IDS.** Dropout MLP (binary head); sklearn Random Forest (120 trees) as
the non-gradient baseline.

**Attacks.** Masked FGSM; 10-step PGD with random start; conditional GAN
trained 24 epochs (UNSW) or 20 epochs (Engelen) against a frozen MLP.
GAN cache is keyed by `(model, eps, epochs)`.

**Metrics.** Clean accuracy / F1 / AUC; evasion rate (among originally
detected attacks); ASR; accuracy drop; mean L2 in scaled space.
Evasion is the primary robustness number.

---

## 5. Experimental setup

**Hardware / software.** This draft’s long runs executed on the project
VM (CPU PyTorch). Seeds `{42, 43, 44}`. Commit hash is the branch that
produced `results/tables/`.

**UNSW-NB15 (official split).**
Training file: 175,341 rows, sha256 `bec7dd5e…0648fa` (Hugging Face
`Mouwiya/UNSW-NB15`). Testing file: 82,332 rows, sha256 `7ec02e7e…cd818c`.
After a 10% val hold-out from train: 157,806 / 17,535 / 82,332, 42
features. Config: `configs/unsw_real_long.yaml`. MLP 10 epochs, batch
256. FGSM/PGD `eps ∈ {0.05, 0.10, 0.15, 0.25}`. GAN 24 epochs at 0.15
and 0.25. Transfer: PGD `eps=0.15` MLP → RF.

**CIC-IDS2017 (Engelen).**
Zip sha256 `4d535da1…ecee0f2a` from
`intrusion-detection.distrinet-research.be/WTMC2021/Dataset/dataset.zip`.
Five day CSVs (row counts from this VM):

| Day | Rows |
| --- | --- |
| Monday | 371,749 |
| Tuesday | 322,003 |
| Wednesday | 496,779 |
| Thursday | 362,368 |
| Friday | 547,915 |
| **Week total** | **2,100,814** |

Full Friday prepare (seed 42, this run): train 383,296 / val 54,757 /
test 109,514 / 78 features after aliasing. Config:
`configs/cicids_engelen_long.yaml`. Official CIC zip was **not** used
(HTML portal).

**Honesty limits.** Friday is PortScan / DDoS / Bot-heavy, not the full
attack taxonomy. The week concat is Engelen-corrected, not official
MachineLearningCSV. One architecture family (MLP) plus RF. Three seeds,
not ten.

---

## 6. Results

Filled only from executed files. Main tables:

- [`results/tables/unsw_real_long_aggregate.md`](../../results/tables/unsw_real_long_aggregate.md)
- [`results/tables/unsw_real_long_matched_l2.md`](../../results/tables/unsw_real_long_matched_l2.md)
- [`results/tables/unsw_real_long_per_seed.md`](../../results/tables/unsw_real_long_per_seed.md)
- Engelen long tables: `results/tables/cicids_engelen_long_*.md` (companion
  sweep; see §6.3 if the run has finished)

Plots (committed copies):
[`results/examples/unsw_real_long_evasion_vs_l2.png`](../../results/examples/unsw_real_long_evasion_vs_l2.png),
[`results/examples/unsw_real_long_attack_comparison.png`](../../results/examples/unsw_real_long_attack_comparison.png),
[`results/examples/unsw_real_long_aggregate_evasion.png`](../../results/examples/unsw_real_long_aggregate_evasion.png),
[`results/examples/unsw_real_long_transfer_heatmap.png`](../../results/examples/unsw_real_long_transfer_heatmap.png).
Earlier single-seed official split: `results/examples/unsw_real_evasion_vs_l2.png`.

### 6.1 Official UNSW-NB15 (3 seeds, MLP)

Mean ± std over seeds 42/43/44. Clean accuracy 0.8644 (F1 on individual
seeds: 0.896 / 0.894 / 0.864). RF clean accuracy 0.9055.

**Table 1.** Constrained attacks on the official test set.

| attack | eps | evasion mean | evasion std | L2 mean | L2 std | acc drop |
| --- | --- | --- | --- | --- | --- | --- |
| FGSM | 0.05 | 0.4542 | 0.1024 | 0.2137 | 0.0068 | 0.2354 |
| FGSM | 0.10 | 0.7301 | 0.2076 | 0.4154 | 0.0142 | 0.3783 |
| FGSM | 0.15 | 0.8451 | 0.1290 | 0.6145 | 0.0216 | 0.4380 |
| FGSM | 0.25 | 0.9215 | 0.0704 | 1.0061 | 0.0361 | 0.4774 |
| PGD | 0.05 | 0.5993 | 0.0323 | 0.2001 | 0.0072 | 0.3111 |
| PGD | 0.10 | 0.8745 | 0.0980 | 0.3770 | 0.0258 | 0.4536 |
| PGD | 0.15 | 0.9305 | 0.0601 | 0.5520 | 0.0519 | 0.4829 |
| PGD | 0.25 | 0.9172 | 0.0712 | 0.8936 | 0.1009 | 0.4759 |
| GAN (24 ep) | 0.15 | 0.3769 | 0.1331 | 0.3275 | 0.0255 | 0.1934 |
| GAN (24 ep) | 0.25 | 0.7215 | 0.0572 | 0.4281 | 0.0095 | 0.3734 |

Source: `unsw_real_long_aggregate.md`.

**Table 2.** Closest matched-L2 GAN vs PGD (and FGSM).

| pair | eps | L2 | evasion | L2 gap |
| --- | --- | --- | --- | --- |
| GAN | 0.15 | 0.3275 | 0.3769 | — |
| PGD (closest) | 0.10 | 0.3770 | 0.8745 | 0.0495 |
| FGSM (closest) | 0.10 | 0.4154 | 0.7301 | 0.0879 |

Source: `unsw_real_long_matched_l2.md`. At nearly the same perturbation
cost, **PGD evades more than twice as often as the 24-epoch GAN**.
Matching `eps=0.15` would have been misleading: GAN’s L2 is closer to
PGD `eps=0.10` than to PGD `eps=0.15` (L2 0.552).

**Table 3.** Transfer (not averaged into Table 1).

| surrogate → target | attack | eps | evasion mean | evasion std | L2 mean |
| --- | --- | --- | --- | --- | --- |
| MLP → RF | PGD | 0.15 | 0.1563 | 0.0851 | 0.5021 |

RF clean accuracy stays high (~0.91). White-box PGD on the MLP is **not**
a remote attacker’s number; 0.16 transfer evasion is the more honest
grey-box estimate on this dump.

### 6.2 Earlier official-split smoke (1 seed, 8-epoch GAN)

`configs/unsw_real.yaml`, seed 42, same files, 8 GAN epochs
(`results/tables/unsw_real_per_seed.md`): clean acc 0.8437; FGSM 0.9851
/ L2 0.6285; PGD 0.9920 / L2 0.5781; GAN 0.1675 / L2 0.4148. Longer GAN
training in §6.1 raised GAN evasion at `eps=0.15` from 0.17 to 0.38
(mean) and at `eps=0.25` to 0.72. Still below matched-L2 PGD.

### 6.3 Engelen CIC-IDS2017

**Prior 20k Friday smoke** (`cicids_engelen_friday`, seed 42,
`results/tables/cicids_engelen_friday_per_seed.md`): clean acc 0.9920;
FGSM/PGD/GAN all evasion 1.000 at `eps=0.15` (L2 0.961 / 0.902 / 0.417).
That sample is PortScan/DDoS-heavy and too small for a ranking.

**Full Friday + week, 3 seeds.** Config `cicids_engelen_long.yaml`.
Friday full split (seed 42 log): 383,296 / 54,757 / 109,514.
Tables from that executed sweep:

<!-- FILLED AFTER cicids_engelen_long COMPLETES -->

_If `results/tables/cicids_engelen_long_aggregate.md` exists, copy its
MLP rows here. Do not invent. Until then this subsection only records
the setup and the 20k smoke._

### 6.4 Negative / unstable cells

GAN evasion on UNSW has a large seed std at `eps=0.15` (0.133). Seed 43
was a weak GAN (log: evasion 0.274 at L2 0.299) while seed 42 reached
0.527. Report the mean **and** the std; do not pick the best seed.
PGD at `eps=0.25` is not strictly stronger than `eps=0.15` (0.917 vs
0.931) — the extra budget does not always help once the mask binds.

---

## 7. Discussion

**Matched L2 matters.** A paper that says “GAN vs PGD at `eps=0.15`”
on this UNSW run would compare L2 0.33 to L2 0.55. The closest-L2 pair
is the fairer one, and it still favours PGD.

**Transfer asymmetry.** MLP→RF PGD evasion 0.16 ± 0.09 on official UNSW
is far below white-box 0.93. A remote attacker with only a forest
target, or only a surrogate, does not get the white-box number.

**Friday-only CIC.** Even the full Friday file is one day (DDoS /
PortScan / Bot). The week concat adds Monday–Thursday but is still
Engelen-corrected CICFlowMeter, not a production sensor.

**Feature space ≠ packets.** A 0.15 shift in `sload` or `Flow Bytes/s`
may be unrealisable. We do not claim CICFlowMeter would emit the
perturbed row.

---

## 8. Limitations

- Three seeds, one MLP width, no CNN / gradient-boosted trees in the
  long tables.
- No TRADES, no certified defense, no problem-space replay.
- CIC official MachineLearningCSV was unreachable (portal). Engelen is
  the correct dump for a paper, but it is not the file most prior work
  cites.
- GAN training still uses a simple AdvGAN loss; 24 epochs is “longer
  than the smoke test,” not a converged WGAN.
- Binary head only.

---

## 9. Conclusion

On official UNSW-NB15, constrained PGD remains the stronger
feature-space evasion against a 10-epoch MLP once cost is measured in
mean L2. A 24-epoch conditional GAN closes some of the gap relative to
an 8-epoch smoke test but does not overtake matched-L2 PGD. Transfer to
Random Forest is an order of magnitude weaker. CIC results belong on
the Engelen dump and must stay labelled Friday-only or week-concat, not
“CIC-IDS2017 SOTA.” The artefacts to rerun are the YAML configs and
`docs/DATA_CARD.md`.

---

## Appendix

- **A.** Hyper-parameters: `results/unsw_real_long/config.json`,
  `results/cicids_engelen_long/config.json`.
- **B.** Per-seed CSV: `results/tables/unsw_real_long_per_seed.csv`.
- **C.** Synthetic protocol check (not a main-table source):
  `results/tables/medium_synthetic_*.md`.
- **D.** Dataset layout and checksums: [DATA_CARD.md](../DATA_CARD.md).

### Reproduction

```bash
python scripts/download_datasets.py
python run.py experiment-suite --config configs/unsw_real_long.yaml
python run.py experiment-suite --config configs/cicids_engelen_long.yaml
```

Do not commit the dumps. Do not paste synthetic `--quick` cells into
Table 1.
