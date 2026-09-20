# Metrics

Every attack evaluation writes JSON under `results/` with the fields below.
Definitions are implemented in `adv_ids/evaluation/metrics.py` and unit-tested
in `tests/test_metrics.py`.

## Clean performance

Computed on the unmodified test split.

| Field | Meaning |
| --- | --- |
| `clean_accuracy` | Share of test flows labelled correctly |
| `precision` / `recall` / `f1` | Binary scores, attack = positive class |
| `roc_auc` | Area under ROC using `P(attack)`; `null` if one class is missing |
| `confusion_matrix` | `[[TN, FP], [FN, TP]]` |

For multiclass training (`--label-mode multiclass`) the same names are used
only when the head is binary. Prefer per-class F1 in the paper write-up.

## Adversarial robustness

Attacks rewrite **attack-labelled** test rows only. Benign rows are copied
into the mixed set unchanged.

| Field | Meaning |
| --- | --- |
| `attack_detected_before` | Attack rows with `ŷ = 1` on the clean vector |
| `attack_detected_after` | Same rows after the perturbation |
| `evasion_rate` | Among *originally detected* attacks, the share that become `ŷ = 0`. This is the primary robustness number. |
| `attack_success_rate` (ASR) | Share of *all* attack rows labelled benign after the attack (includes originally missed attacks) |
| `adversarial_accuracy` | Accuracy on the mixed set (benign clean + attacks replaced) |
| `accuracy_drop` | `clean_accuracy − adversarial_accuracy` |

`evasion_rate` and ASR answer different questions. A detector with poor clean
recall can look “robust” on ASR (many attacks were already missed) while
`evasion_rate` stays honest about the ones it used to catch. **Report both.**

## Perturbation cost

Measured in **min-max scaled** feature space.

| Field | Meaning |
| --- | --- |
| `mean_l2_perturbation` | Mean Euclidean norm of `x' − x` |
| `median_l2_perturbation` | Median of the same |
| `mean_linf_perturbation` | Mean L∞ norm |
| `mean_abs_perturbation` | Mean absolute per-coordinate change |
| `max_frozen_feature_change` | Should be ~0 if the mask is applied |

An attack that evades with L2 ≈ 0.01 is more interesting than one that dumps
the vector on the opposite corner of the unit cube. Always pair ASR with L2.

## Protocol (what to put in a paper table)

For each `(dataset, model, attack, defense)` cell:

1. Train with a fixed seed; log the config JSON next to the metrics.
2. Report clean accuracy / F1 / AUC.
3. Craft adversarial attack rows (`eps`, steps, mask documented).
4. Report evasion rate, ASR, accuracy drop, mean L2.
5. Repeat seeds if you claim a ranking (3 seeds is a minimum).

Transfer cells add `surrogate → target` and should not be averaged with
white-box cells.

## What is *not* a result

- Metrics from the synthetic generator, unless clearly labelled “schema smoke
  test / CI”.
- Empty template tables. `results/tables/` holds headers only until a run
  writes numbers.
- A single unseeded notebook cell.

The smoke pipeline writes `results/evaluation_metrics.json` from whatever you
actually ran. The experiment suite writes `results/<run_name>/suite_metrics.md`.
