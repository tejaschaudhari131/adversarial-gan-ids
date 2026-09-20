| dataset | model | defense | eps | n_seeds | attacks | fgsm_evasion | fgsm_l2 | fgsm_acc_drop | gan_evasion | gan_l2 | gan_acc_drop | pgd_evasion | pgd_l2 | pgd_acc_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cicids2017 | mlp | none | 0.1500 | 3 | fgsm,gan,pgd | 0.9900 | 1.1975 | 0.3300 | 0.9850 | 1.1764 | 0.3283 | 0.9900 | 1.1970 | 0.3300 |
| cicids2017 | mlp | none | 0.2500 | 3 | fgsm,gan,pgd | 1.0000 | 1.9814 | 0.3333 | 1.0000 | 1.6646 | 0.3333 | 1.0000 | 1.9125 | 0.3333 |
| cicids2017 | mlp_advtrain | adv_train | 0.1500 | 3 | fgsm,pgd | 0.2887 | 1.1973 | 0.0950 |  |  |  | 0.2871 | 1.1901 | 0.0944 |
| cicids2017 | mlp_advtrain | adv_train | 0.2500 | 3 | fgsm,pgd | 0.6213 | 1.9809 | 0.2044 |  |  |  | 0.5944 | 1.8970 | 0.1956 |
| unsw_nb15 | mlp | none | 0.1500 | 3 | fgsm,gan,pgd | 0.9317 | 0.8609 | 0.3106 | 0.9133 | 0.8499 | 0.3044 | 0.9317 | 0.8606 | 0.3106 |
| unsw_nb15 | mlp | none | 0.2500 | 3 | fgsm,gan,pgd | 1.0000 | 1.4263 | 0.3333 | 1.0000 | 1.2486 | 0.3333 | 1.0000 | 1.3773 | 0.3333 |
| unsw_nb15 | mlp_advtrain | adv_train | 0.1500 | 3 | fgsm,pgd | 0.6433 | 0.8609 | 0.2144 |  |  |  | 0.6433 | 0.8609 | 0.2144 |
| unsw_nb15 | mlp_advtrain | adv_train | 0.2500 | 3 | fgsm,pgd | 0.9983 | 1.4263 | 0.3328 |  |  |  | 0.9933 | 1.3760 | 0.3311 |
