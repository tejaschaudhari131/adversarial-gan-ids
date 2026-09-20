# Results layout

Nothing in this folder is a published CIC-IDS2017 number unless the JSON
`dataset` field says so **and** `synthetic` is false.

```
results/
  evaluation_metrics.json          # last `run.py pipeline` (historical filename)
  robustness_summary.png
  confusion_matrix_clean.png
  roc_curve_clean.png
  <suite_name>/
    config.json                    # hyperparameters actually used
    suite_metrics.json
    suite_metrics.md
    attack_comparison.png
    transfer_heatmap.png
    <dataset>/<model>/evaluation_<attack>.json
  tables/
    template_attack_comparison.md  # headers only; fill from a real run
```

The experiment suite refuses to invent cells: if an attack is skipped (e.g.
FGSM on Random Forest) the row is omitted, not zero-filled.
