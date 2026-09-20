# Threat model

This document states what the adversary is allowed to know and change. The
implementation is a **research evaluation tool on flow-feature vectors**, not a
packet-crafting exploit kit. A successful evasion here means “the IDS score on
a *modified feature vector* crossed the decision threshold,” not “this payload
is guaranteed to survive a real NIC, middlebox, and CICFlowMeter.”

The taxonomy follows Apruzzese et al., “Modeling Realistic Adversarial Attacks
against Network Intrusion Detection Systems,” *Digital Threats: Research and
Practice* 3(3), 2022 ([DOI 10.1145/3469659](https://doi.org/10.1145/3469659);
[arXiv:2106.09380](https://arxiv.org/abs/2106.09380)): training-data access,
feature-set knowledge, detector knowledge, oracle access, and manipulation
depth.

## Goal

**Evasion (test-time integrity).** The attacker already has a malicious flow
that the IDS would flag. They add a *budgeted, masked* perturbation so the
detector labels the flow benign, while frozen protocol-semantic features
(ports, flags, protocol IDs) stay put.

Poisoning, availability floods against the sensor, and problem-space packet
mutation (Pierazzi et al., IEEE S&P 2020) are out of scope for the current
code.

## Knowledge settings

| Setting | What the attacker has | What this repo implements |
| --- | --- | --- |
| **White-box** | Architecture, weights, and gradients of the target IDS | Constrained FGSM and PGD against MLP / deep MLP / 1D-CNN |
| **Grey-box (AdvGAN-style)** | A frozen differentiable IDS with the same feature space; query + gradient access during GAN training, not necessarily after | Conditional GAN `G(x, z)` trained against a frozen IDS |
| **Transfer / surrogate black-box** | A locally trained substitute; no gradients of the target | Craft on the surrogate (PGD or GAN), evaluate on Random Forest / another net |

The default GAN pipeline is **grey-box**: the IDS is frozen, the generator sees
its `P(attack)`, and protocol features are masked. That matches the AdvGAN
semi-white-box recipe (Xiao et al., IJCAI 2018) more closely than a
query-only commercial IDS.

## Capability (what may be changed)

All attacks share the same **domain constraints**:

1. **Feature mask.** Destination port, TCP flag counts, protocol identifiers,
   and related binary flags are frozen (`docs` + `adv_ids/data/schemas.py`).
   Timing, sizes, rates, window, and idle/active statistics may move.
2. **Box constraints.** Features live in the scaled cube `[0, 1]` after
   min-max normalisation. Perturbations are clipped back into that cube.
3. **Budget.** FGSM/PGD use an L∞ radius `eps` (default `0.15`–`0.25` in
   scaled space). The GAN uses `tanh` outputs times `eps`, plus an L2 penalty.
4. **Label-preserving intent.** Only attack rows are perturbed. Benign test
   rows stay clean so accuracy drop is not an artefact of rewriting normals.

These constraints follow the “restricted modification” idea in IDSGAN (Lin,
Shi, Xue, PAKDD 2022) and the constrained-domain discussion of Sheatsley et
al. (JCS 2022 / arXiv:2011.01183). They are **necessary but not sufficient**
for a problem-space flow: a 0.2 shift in `Flow Bytes/s` may not correspond to
any sequence of packets that CICFlowMeter would emit.

## Defender

- Trains on labelled flow features (binary by default).
- May deploy **adversarial training** (PGD on attack rows, mixed with clean
  batches) as the first defense baseline.
- Does not get to see the attacker’s generator at test time.

## What we do *not* claim

- Real-time packet injection, checksum-correct TCP, or bypass of a deployed
  Suricata/Snort/CICFlowMeter stack.
- That synthetic-schema numbers generalise to CIC-IDS2017/2018.
- That white-box PGD is a realistic remote attacker. It is an **upper bound**
  on feature-space vulnerability; transfer numbers are the more honest
  grey-box estimate.

## Recommended reporting

When you publish, state for each row of the table: dataset + correction status
(official vs. Engelen-cleaned), target model, attack family, `eps`, mask,
threat-model cell (white-box / grey-box / transfer), and whether the defense
was on. See [METRICS.md](METRICS.md).
