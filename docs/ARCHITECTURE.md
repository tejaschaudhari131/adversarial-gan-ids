# Code architecture

```
run.py                         CLI (prepare | train-ids | train-attack | evaluate |
                               pipeline | experiment-suite)
adv_ids/
  data/                        schemas, synthetic generators, loaders, masks, splits
  models/                      shared IDSModel: MLP, deep MLP, 1D-CNN, Random Forest, GAN
  attacks/                     feature constraints, FGSM, PGD, GAN, transfer helper
  defenses/                    PGD adversarial training
  training/                    fit / load wrappers (incl. legacy ids_model.pt)
  evaluation/                  metrics, plots, GAN-vs-MLP protocol
  experiments/                 YAML-driven suite
  utils/                       seed, config, JSON I/O
configs/                       quick.yaml, full.yaml, multi_dataset.yaml
docs/                          research plan, threat model, metrics, related work
tests/                         preprocess, masks, metrics, attacks, configs
```

Legacy import paths (`preprocessing`, `models`, `training`, `evaluation`) re-export
the new package so older scripts keep working.

Attacks take an `IDSModel`. Differentiable models expose
`attack_logits_torch`; Random Forest does not — use a surrogate (see transfer).
