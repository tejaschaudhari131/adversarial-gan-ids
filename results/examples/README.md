# Example metrics from runs that actually executed

These files were produced on this machine by:

```bash
python run.py pipeline --synthetic --quick
python run.py experiment-suite --config configs/quick.yaml
```

They are **synthetic-schema smoke tests**, not CIC-IDS2017/UNSW paper results.
Do not copy them into a publication table.

| File | Source |
| --- | --- |
| `pipeline_synthetic_quick.json` | default GAN-vs-MLP pipeline (`eps=0.25`, 4 IDS + 8 GAN epochs) |
| `suite_synthetic_quick.md` | two-dataset sweep (MLP, RF, FGSM, PGD, GAN, adv-training, transfer) |
| `suite_synthetic_quick.json` | same rows as JSON |
| `suite_synthetic_quick_config.json` | hyperparameters of that suite |

Observations from *this* smoke run (not general claims):

- The default pipeline GAN evaded 98.8% of detected attacks at `eps=0.25`.
- In the suite, constrained FGSM/PGD evaded the MLP; the 8-epoch GAN at
  `eps=0.15` did not (mean L2 stayed smaller). That is a training-budget
  result, not a proof that PGD dominates GAN.
- Transfer PGD (MLP → Random Forest) was weak on the CIC stand-in (5.8%)
  and strong on the UNSW stand-in (77.5%). Asymmetry is expected; measure
  it on real dumps before writing it down.
