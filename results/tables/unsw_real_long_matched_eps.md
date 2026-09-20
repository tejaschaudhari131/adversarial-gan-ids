| dataset | model | defense | eps | n_seeds | attacks | fgsm_evasion | fgsm_l2 | fgsm_acc_drop | pgd_evasion | pgd_l2 | pgd_acc_drop | gan_evasion | gan_l2 | gan_acc_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unsw_nb15 | mlp | none | 0.0500 | 3 | fgsm,pgd | 0.4542 | 0.2137 | 0.2354 | 0.5993 | 0.2001 | 0.3111 |  |  |  |
| unsw_nb15 | mlp | none | 0.1000 | 3 | fgsm,pgd | 0.7301 | 0.4154 | 0.3783 | 0.8745 | 0.3770 | 0.4536 |  |  |  |
| unsw_nb15 | mlp | none | 0.1500 | 3 | fgsm,gan,pgd | 0.8451 | 0.6145 | 0.4380 | 0.9305 | 0.5520 | 0.4829 | 0.3769 | 0.3275 | 0.1934 |
| unsw_nb15 | mlp | none | 0.2500 | 3 | fgsm,gan,pgd | 0.9215 | 1.0061 | 0.4774 | 0.9172 | 0.8936 | 0.4759 | 0.7215 | 0.4281 | 0.3734 |
