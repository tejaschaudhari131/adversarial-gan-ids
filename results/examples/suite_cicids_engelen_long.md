# Engelen CIC-IDS2017 long suite (full Friday + week, 3 seeds)

Source tables: `results/tables/cicids_engelen_long_*.md`.
Config: `configs/cicids_engelen_long.yaml`. Seeds 42/43/44. Official
MachineLearningCSV was not used (HTML portal).

Friday: 547,915 raw → train 383,296 / val 54,757 / test 109,514, 78 features.
Week concat: 2,100,814 raw → 1,470,014 / 210,002 / 420,005.

MLP clean accuracy: Friday 0.9968, week 0.9896.
RF clean accuracy: Friday 0.9973, week 0.9917.

Closest matched-L2 (`cicids_engelen_long_matched_l2.md`):

| slice | GAN eps / L2 / ev | PGD eps / L2 / ev | L2 gap |
| --- | --- | --- | --- |
| Friday | 0.25 / 0.5417 / 0.9972 | 0.10 / 0.6022 / 0.9465 | 0.0605 |
| Week | 0.15 / 0.3397 / 0.9555 | 0.05 / 0.2757 / 0.8905 | 0.0640 |

Friday FGSM evasion is flat at ~0.716 across `eps ∈ {0.05,0.10,0.15,0.25}`.
Transfer MLP→RF PGD 0.15: Friday 0.9937, week 0.9982.

These cells are from the executed suite only. Friday is still one day
(PortScan / DDoS / Bot). The week file is Engelen-corrected, not official CIC.
