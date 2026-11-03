# Repository Guide

## Repository structure

| Location | Contents |
|---|---|
| `src/` | Deep SAD implementation, original components, and project adaptations |
| `src/datasets/` | Original and extension dataset loaders |
| `src/networks/` | Encoder and autoencoder architectures |
| `src/optim/` | Training and evaluation routines |
| `scripts/` | Dataset generation, experiment automation, and visualisation |
| `environment.yml` | Project software environment |
| `docs/contributions.md` | Attribution and project contributions |
| `reports/project-update/` | Historical presentation-stage documentation and pilot evidence |
| `results/experiments-50ep-ae100/` | Earlier constructed evaluation that discarded incomplete batches |
| `results/final-full-batches/` | Corrected constructed evaluation retaining incomplete batches |
| `data/` | Locally downloaded and generated datasets |

See `contributions.md` for the distinction between inherited, adapted, and added components.

## Environment setup

Run commands from the repository root.

```bash
conda env create -f environment.yml
conda activate deep-sad
```

The project uses Python 3.10 and CPU versions of PyTorch and torchvision.
Use `environment.yml` rather than the upstream legacy `requirements.txt`.

## Dataset preparation

The MNIST loader downloads the original dataset when required.
The GalaxyMNIST loader downloads its catalogues and image archive, checks the downloads, and resizes images to 32 x 32 RGB.

Generate the constructed dataset with:

```bash
python scripts/generate_corrupted_mnist.py \
  --max-samples 1000 \
  --severity 1 \
  --seed 1
```

This selects 1,000 source images from each original MNIST split, with 100 images per digit. Each source produces four versions:

- Clean
- Rotation by either -15 or +15 degrees
- Gaussian noise with standard deviation 0.15
- Black square occlusion of 6 x 6 pixels

The generator writes `train.pt`, `test.pt`, `metadata.csv`, and
`summary.json` to `data/constructed_mnist/`.

Running generation again overwrites these generated files.
Preserve the same generation settings when reproducing the results.

## Constructed MNIST protocol

Digit 0 is normal; digits 1-9 are anomalous. Labelled anomalies are selected from digit 1.

Each image condition contains 1,000 observations per split.
The semi-supervised training subset contains:

- 100 unlabelled normal observations in the 0% condition.
- 100 unlabelled normal observations and 5 labelled anomalies in the
  nominal 5% condition.

The realised labelled fraction is 5/105, approximately 4.76%, because sample counts are rounded down to integers.

Each test condition contains 100 normal and 900 anomalous observations.
Every model is trained and tested in the same image condition.

Generation uses a fixed seed. Training seeds 1, 2, and 3 vary model initialisation and training-sample selection; they do not generate three independent source datasets.

The corrected constructed loader retains incomplete batches. With batch size 64, training batches have sizes 64 and 36, or 64 and 41.

## Reproduce the corrected constructed experiments

Use a separate output root for reproduction so the committed evidence remains unchanged.

Run the 18 corruption experiments:

```bash
python scripts/run_experiments.py \
  --suite constructed \
  --n-epochs 50 \
  --ae-n-epochs 100 \
  --device cpu \
  --batch-size 64 \
  --results-root results/reproduction-full-batches
```

Run the six clean-control experiments:

```bash
python scripts/run_experiments.py \
  --suite constructed_clean \
  --n-epochs 50 \
  --ae-n-epochs 100 \
  --device cpu \
  --batch-size 64 \
  --results-root results/reproduction-full-batches
```

The combined evaluation contains 24 runs: four conditions, two labelled-anomaly ratios, and three training seeds.

The runner specifies epoch counts, batch size, seed, classes, pollution, and image condition. Other training settings use the defaults in `src/main.py`, including learning rates of 0.001, weight decay of 1e-6 for both stages, and eta of 1.0. The resulting `config.json` records the run settings.

## Reproduce the pilot experiments

MNIST pilot, 18 runs:

```bash
python scripts/run_experiments.py \
  --suite mnist \
  --n-epochs 5 \
  --ae-n-epochs 5 \
  --device cpu \
  --batch-size 128 \
  --results-root results/reproduction-pilots
```

GalaxyMNIST pilot, six runs:

```bash
python scripts/run_experiments.py \
  --suite galaxy \
  --n-epochs 5 \
  --ae-n-epochs 5 \
  --device cpu \
  --batch-size 128 \
  --results-root results/reproduction-pilots
```

These pilots have a different training budget from the corrected constructed experiments. Their absolute AUC values should not be interpreted as a controlled comparison between datasets.

The MNIST and GalaxyMNIST loaders retain the upstream behaviour of discarding incomplete training batches.

## Inspect commands before execution

Add `--dry-run` to a runner command to print the planned commands.
Dry runs do not train models or write summary files.

The `all` suite includes MNIST, GalaxyMNIST, constructed corruptions, and the clean control. It applies the same requested epoch counts and batch size to every suite. Use the separate commands above to reproduce the different pilot and constructed training budgets.

## Saved outputs and resuming

Individual run directories contain:

- `config.json`: experiment settings.
- `results.json`: Deep SAD AUC, timings, and indexed test scores.
- `ae_results.json`: autoencoder evaluation and timings.

Training also generates logs and model files locally. The committed result evidence contains JSON and CSV files rather than trained weights.

The runner writes a summary CSV under the selected results root.
Existing runs are checked against the requested configuration before being reused. A configuration mismatch stops execution.

Use a new results root after changing the data, code, or experimental protocol: the configuration check does not detect every possible code or dataset change.

The `--force` option reruns experiments in existing directories and can overwrite their outputs.

## Figures and historical evidence

Scripts beginning with `plot_` or `visualize_` generate analysis figures.
Check their input and output paths before running them: some were originally written for presentation-stage results and may still point to earlier experiment directories.

The corrected constructed result source is:

```text
results/final-full-batches/
```

Presentation-stage figures must be regenerated from the corrected results before being used to describe the final evaluation.

## Scope and outstanding work

The repository contains MNIST and GalaxyMNIST pilots and the corrected constructed evaluation. The proposed screening of five external dataset candidates was not part of the verified completed experiments.

LLM-based quality control remains pending at this documentation stage.
Its procedure and evidence will be added when completed.