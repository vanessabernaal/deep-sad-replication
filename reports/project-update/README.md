# Deep SAD Project Update

## Project Overview

This project partially replicates and extends Deep Semi-Supervised Anomaly Detection (Deep SAD), introduced by Ruff et al. at ICLR 2020. Deep SAD learns a compact representation of normal observations while using a limited number of labelled anomalies to improve anomaly detection.

The project investigates whether labelled anomalies improve ROC-AUC:

1. on a reduced replication of the original MNIST experiment;
2. in a new scientific image domain using GalaxyMNIST;
3. for systematically corrupted versions of MNIST.

The implementation is based on the authors' official PyTorch repository.

## Computational Environment

The project is running in GitHub Codespaces on CPU using:

- Python 3.10;
- PyTorch 2.2.2;
- torchvision 0.17.2;
- NumPy 1.26.4;
- scikit-learn 1.4.2.

An `environment.yml` file records the required software. Compatibility updates were made for modern versions of NumPy, Click, PyTorch, and torchvision.

## Replication Progress

The original implementation now runs successfully on MNIST. A reduced pilot experiment evaluated:

- normal digits 0 and 3;
- labelled-anomaly ratios of 0%, 1%, and 5%;
- random seeds 1, 2, and 3;
- five autoencoder pretraining epochs;
- five Deep SAD training epochs.

Mean test ROC-AUC results were:

| Normal class | 0% labelled | 1% labelled | 5% labelled |
|---|---:|---:|---:|
| Digit 0 | 96.20% | 97.27% | 97.77% |
| Digit 3 | 86.15% | 89.00% | 89.87% |

The pilot results support the paper's central claim: limited labelled anomalies improved detection performance for both selected normal classes.

## Existing New Dataset: GalaxyMNIST

GalaxyMNIST introduces a scientific image domain containing four galaxy morphology classes:

- smooth round;
- smooth cigar-shaped;
- edge-on disk;
- unbarred spiral.

The downloaded dataset contains 8,000 training and 2,000 test images. Original RGB images have a resolution of 224 by 224 pixels and are resized to 32 by 32 pixels for compatibility with the CIFAR-10 architecture.

The GalaxyMNIST pilot used class 0 as normal and compared 0% and 5% labelled anomalies over three random seeds.

| Labelled anomalies | Mean ROC-AUC | Standard deviation |
|---:|---:|---:|
| 0% | 64.59% | 0.37 |
| 5% | 73.03% | 0.56 |

Adding 5% labelled anomalies improved mean ROC-AUC by approximately 8.44 percentage points.

## Constructed Dataset

A controlled-corruption version of MNIST was generated programmatically. For each selected source image, four observations were created:

- one clean image;
- rotation by 15 degrees;
- Gaussian noise with standard deviation 0.15;
- square occlusion of 6 by 6 pixels.

Training and test images were generated from their original MNIST partitions to prevent data leakage. The prototype uses 100 source images from each partition, producing 400 training and 400 test observations.

Metadata records:

- original MNIST index;
- digit label;
- data split;
- corruption type;
- severity;
- transformation parameters;
- random seed.

Validation checks confirm image dimensions, value ranges, observation counts, metadata consistency, and separation of training and test sources.

## Constructed Dataset Pilot Results

Each corruption was evaluated independently using:

- 100 clean training images;
- either 0 or 5 labelled corrupted training images;
- 100 clean and 100 corrupted test images;
- three random seeds;
- five pretraining and five Deep SAD training epochs.

| Corruption | 0% labelled | 5% labelled |
|---|---:|---:|
| Rotation | 48.69% | 48.56% |
| Gaussian noise | 39.34% | 40.05% |
| Square occlusion | 55.84% | 55.38% |

Under this small pilot configuration, adding labelled anomalies did not produce a consistent improvement. The result suggests that the benefit observed for class-based anomalies may not transfer directly to controlled visual corruptions. However, the prototype size, short training duration, and variability across seeds prevent a definitive conclusion.

## Reproducibility

The repository now includes:

- a reproducible Conda environment;
- scripts for generating constructed MNIST;
- loaders for GalaxyMNIST and constructed MNIST;
- a resumable experiment-suite runner;
- scripts for producing result figures;
- configuration files, logs, summary CSV files, and representative examples.

Raw datasets and complete experiment outputs are excluded from Git because of their size. The scripts recreate the datasets and experiments.

## Current Challenges and Next Steps

Completed:

- original MNIST implementation operational;
- reduced MNIST replication;
- GalaxyMNIST integration and pilot;
- constructed MNIST generation and validation;
- corruption pilot experiments;
- quantitative and qualitative visualisations.

Next steps:

1. perform LLM-based quality control on a stratified image sample;
2. expand the constructed dataset beyond the 100-image prototype;
3. evaluate additional corruption severities if resources permit;
4. run longer final experiments;
5. complete the final report and reproducibility instructions.