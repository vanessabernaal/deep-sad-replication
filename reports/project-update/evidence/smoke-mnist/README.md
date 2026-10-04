# MNIST smoke test

This smoke test verifies that the original Deep SAD workflow runs successfully in the modernised CPU environment.

## Configuration

- Dataset: MNIST
- Network: mnist_LeNet
- Normal class: 0
- Known anomaly class: 1
- Labelled-anomaly ratio: 5%
- Random seed: 1
- Autoencoder pretraining epochs: 1
- Deep SAD training epochs: 1
- Device: CPU

## Results

- Autoencoder test ROC-AUC: 72.64%
- Deep SAD test ROC-AUC: 95.21%
- Test observations: 10,000
- Autoencoder training time: 4.81 seconds
- Deep SAD training time: 1.56 seconds

This is a compatibility and feasibility test, not a final replication result. The final experiments will use the planned configurations and repeated random seeds.
