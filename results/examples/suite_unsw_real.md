| seed | dataset | model | attack | defense | eps | clean_accuracy | adversarial_accuracy | accuracy_drop | evasion_rate | attack_success_rate | mean_l2_perturbation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | unsw_nb15 | mlp | fgsm | none | 0.1500 | 0.8437 | 0.3369 | 0.5067 | 0.9851 | 0.9861 | 0.6285 |
| 42 | unsw_nb15 | mlp | pgd | none | 0.1500 | 0.8437 | 0.3333 | 0.5103 | 0.9920 | 0.9926 | 0.5781 |
| 42 | unsw_nb15 | mlp | gan | none | 0.1500 | 0.8437 | 0.7712 | 0.0725 | 0.1675 | 0.1973 | 0.4148 |
