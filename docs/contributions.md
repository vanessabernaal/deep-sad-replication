# Project Contributions and Attribution

## Original research and implementation

This project builds on Deep Semi-Supervised Anomaly Detection (Deep SAD), introduced by Ruff et al. at ICLR 2020.

- Paper: https://arxiv.org/abs/1906.02694
- Original code: https://github.com/lukasruff/Deep-SAD-PyTorch

The Deep SAD method, original neural network architectures, training objectives, baseline implementations, and original dataset support were provided by the upstream authors.

The original MIT licence and attribution are retained.

## Work carried out in this project

The following contributions form part of Vanessa Fernanda Bernal Hernandez's COMP8240 replication and extension project. They extend the existing implementation rather than introduce a new detection method.

| Component | Project contribution | Main files |
|---|---|---|
| Software environment | Defined a modern CPU environment and adapted compatibility-sensitive code | `environment.yml`, adapted files in `src/` |
| GalaxyMNIST | Integrated catalogue loading, image preparation, resizing, and the semi-supervised setting | `src/datasets/galaxy_mnist.py` |
| Constructed MNIST | Implemented stratified source selection, controlled transformations, metadata, validation, and example generation | `scripts/generate_corrupted_mnist.py` |
| Constructed dataset loader | Selected image conditions and derived anomaly labels from digit identity | `src/datasets/constructed_mnist.py` |
| Dataset integration | Connected the additional datasets and corruption option to the application | `src/datasets/main.py`, `src/main.py` |
| Experiment automation | Implemented experiment suites, output directories, summaries, and configuration checks when reusing runs | `scripts/run_experiments.py` |
| Analysis and visualisation | Added scripts for summarising results and inspecting representative images | Project analysis scripts in `scripts/` |
| Experimental evidence | Recorded pilot and corrected experiment outputs, configurations, and figures | `reports/project-update/`, `results/final-full-batches/` |
| Documentation | Documented the project scope, contributions, environment, and reproduction workflow | `README.md`, `docs/` |

## Adaptations to original components

Files inherited from the upstream repository may contain project modifications. Their location in `src/` does not imply that they were written entirely for this project.

The core distinction is:

- **Inherited:** Deep SAD methodology and the original implementation.
- **Adapted:** compatibility updates and integration with the new datasets.
- **Added:** extension loaders, constructed-data generation, experiment
  automation, analysis scripts, and project documentation.

Git history records the changes made during project development.

## Dataset attribution

MNIST and GalaxyMNIST are externally created datasets. This project does not claim authorship of their source images or original labels.

Constructed MNIST is a derived dataset: its contribution consists of source selection, controlled transformations, metadata, validation, and the experimental protocol. The original digit labels are retained.

## Experimental revisions

Earlier constructed experiments explored a different anomaly definition and were superseded by the digit-based design.

Under the corrected definition:

- Digit 0 is normal.
- Digits 1-9 are anomalous.
- Labelled anomalies come from digit 1.
- Clean and corrupted versions retain the same digit-based labels.

The evaluation in `results/experiments-50ep-ae100/` discarded incomplete training batches. The corrected evaluation in `results/final-full-batches/` retains them and uses every selected training observation in each epoch.

Earlier evidence is retained to document development. It should not be combined with the corrected results as if it used the same protocol.