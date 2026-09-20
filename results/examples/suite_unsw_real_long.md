# Official UNSW-NB15 long suite (3 seeds)

Source tables: `results/tables/unsw_real_long_*.md`.
Config: `configs/unsw_real_long.yaml`. Official 175,341 / 82,332 split;
scaler + categoricals fit on train only. MLP 10 epochs, GAN 24 epochs,
seeds 42/43/44.

Mean MLP clean accuracy 0.8644. RF clean accuracy 0.9055.

| attack | eps | evasion mean | evasion std | L2 mean |
| --- | --- | --- | --- | --- |
| FGSM | 0.05 | 0.4542 | 0.1024 | 0.2137 |
| FGSM | 0.10 | 0.7301 | 0.2076 | 0.4154 |
| FGSM | 0.15 | 0.8451 | 0.1290 | 0.6145 |
| FGSM | 0.25 | 0.9215 | 0.0704 | 1.0061 |
| PGD | 0.05 | 0.5993 | 0.0323 | 0.2001 |
| PGD | 0.10 | 0.8745 | 0.0980 | 0.3770 |
| PGD | 0.15 | 0.9305 | 0.0601 | 0.5520 |
| PGD | 0.25 | 0.9172 | 0.0712 | 0.8936 |
| GAN (24 ep) | 0.15 | 0.3769 | 0.1331 | 0.3275 |
| GAN (24 ep) | 0.25 | 0.7215 | 0.0572 | 0.4281 |
| transfer MLP→RF PGD | 0.15 | 0.1563 | 0.0851 | 0.5021 |

Closest matched-L2 pair (`unsw_real_long_matched_l2.md`): GAN `eps=0.15`
(L2 0.3275, evasion 0.3769) vs PGD `eps=0.10` (L2 0.3770, evasion 0.8745),
L2 gap 0.0495.

These cells are from the executed suite only.
