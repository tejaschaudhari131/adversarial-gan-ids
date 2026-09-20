# Code architecture

```
run.py                         CLI (prepare | train-ids | train-attack | evaluate |
                               pipeline | experiment-suite | check-data | setup-data)
adv_ids/
  data/                        schemas, catalog, setup/fetch, synthetic, loaders, masks
  models/                      shared IDSModel: MLP, deep MLP, 1D-CNN, Random Forest, GAN
  attacks/                     feature constraints, FGSM, PGD, GAN, transfer helper
  defenses/                    PGD adversarial training
  training/                    fit / load wrappers (incl. legacy ids_model.pt)
  evaluation/                  metrics, plots, GAN-vs-MLP protocol
  experiments/                 YAML-driven suite + multi-seed aggregate tables
  utils/                       seed, config, JSON I/O
configs/                       quick.yaml, medium.yaml, unsw_real.yaml, unsw_real_long.yaml,
                               cicids_engelen_friday.yaml, cicids_engelen_long.yaml, full.yaml
docs/                          research plan, paper outline, paper/DRAFT.md, threat model, metrics, datasets
tests/                         preprocess, masks, metrics, attacks, fixtures, catalog
```

Legacy import paths (`preprocessing`, `models`, `training`, `evaluation`) re-export
the new package so older scripts keep working.

Attacks take an `IDSModel`. Differentiable models expose
`attack_logits_torch`; Random Forest does not — use a surrogate (see transfer).
