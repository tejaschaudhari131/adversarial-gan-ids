# Adversarial GAN Against Intrusion Detection Systems

A working PyTorch pipeline that trains a deep-learning IDS on **CIC-IDS2017-style flow features**, then trains a **conditional adversarial GAN** to perturb attack flows so the IDS labels them benign.

This is a rebuild of [tejaschaudhari131/adversarial-gan-ids](https://github.com/tejaschaudhari131/adversarial-gan-ids). The original tree was a TensorFlow sketch: the GAN loaded CSVs with `np.loadtxt` (breaks on CIC-IDS2017 headers and labels), the generator dimension was hard-coded to 128 (CIC-IDS2017 has 78 flow features), and evaluation never computed an evasion rate. This version trains, attacks, and reports **evasion rate** and **accuracy drop**.

## What it does

1. Load a real CIC-IDS2017 `MachineLearningCSV` file, **or** generate a same-schema synthetic stand-in so the repo runs without the multi-GB download.
2. Clean inf/NaN values, min-max scale features to `[0, 1]`, binarize labels (`BENIGN=0`, any attack=`1`).
3. Train a dropout MLP IDS.
4. Freeze the IDS. Train a generator `G(x_attack, z)` that adds a **masked, budgeted perturbation**. Frozen features (destination port, TCP flags) cannot move. The generator is trained to:
   - look like real benign traffic (discriminator)
   - drive IDS `P(attack)` toward 0
   - keep the L2 perturbation small
5. Evaluate:
   - clean accuracy / precision / recall / F1 / ROC-AUC
   - **evasion rate** = share of originally detected attacks now classified as benign
   - **adversarial accuracy** and **accuracy drop** on the mixed test set

## Quick start (no CIC-IDS2017 download)

```bash
python -m pip install -r requirements.txt
python run.py pipeline --synthetic --quick
```

`--quick` uses a smaller table and fewer epochs. A full synthetic run:

```bash
python run.py pipeline --synthetic
```

Artifacts land in `artifacts/` (scaler, IDS, GAN). Plots and `evaluation_metrics.json` land in `results/`.

## Use the real CIC-IDS2017 dataset

1. Download **MachineLearningCSV.zip** from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Unzip a day file (for example `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`).
3. Run:

```bash
python run.py pipeline --dataset path/to/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Column names are stripped automatically. Non-numeric columns are dropped. Any label other than `BENIGN` is treated as attack.

Individual stages:

```bash
python run.py prepare   --dataset data/cicids2017.csv
python run.py train-ids --dataset data/cicids2017.csv --ids-epochs 12
python run.py train-gan --dataset data/cicids2017.csv --gan-epochs 40 --eps 0.15
python run.py evaluate  --dataset data/cicids2017.csv
```

## Project layout

```
run.py                         # CLI: prepare | train-ids | train-gan | evaluate | pipeline
preprocessing/data_preprocessing.py
models/ids_model.py            # binary MLP IDS
models/generator.py            # G(x, z) -> bounded perturbation
models/discriminator.py
training/train_ids.py
training/train_gan.py          # fool-IDS + look-benign + small-delta
evaluation/evaluate_model.py   # evasion rate and accuracy drop
```

## Metrics that matter

| Metric | Meaning |
| --- | --- |
| Clean accuracy | IDS on unmodified test flows |
| Adversarial accuracy | Same labels, attack rows replaced by GAN outputs |
| Accuracy drop | Clean minus adversarial accuracy |
| Evasion rate | Detected attacks that become false negatives after perturbation |
| Mean L2 perturbation | Size of the change in scaled feature space |

## Design notes

- Features are scaled to `[0, 1]`. Perturbations are `tanh` outputs times `--eps` (default `0.25`), then clipped back into the unit cube.
- A **modifiable-feature mask** freezes destination port and TCP flag counts so the attack cannot rewrite protocol semantics. Timing, sizes, rates, and window features can move.
- The IDS is **frozen** while the GAN trains. This is a grey-box evasion setting: the attacker has query/gradient access to a substitute IDS with the same architecture as the target.
- This is a **research evaluation tool**, not an exploit kit. It operates on already-extracted flow feature vectors, not live packets.

## Resume-style summary

Developed a Python/PyTorch adversarial GAN on CIC-IDS2017-style traffic to test deep-learning IDS robustness. The generator produces budgeted, feature-constrained perturbations of attack flows. Robustness is measured by evasion rate and accuracy degradation under adversarial input.

## License

MIT. See [LICENSE](LICENSE).
