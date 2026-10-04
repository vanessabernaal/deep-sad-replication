# Constructed MNIST prototype

This prototype verifies the programmatic generation of controlled MNIST corruptions.

## Prototype configuration

- Random seed: 1
- Source images: 100 training and 100 test images
- Sampling: stratified across the ten digit classes
- Severity level: 1
- Rotation: 15 degrees in either direction
- Gaussian-noise standard deviation: 0.15
- Square-occlusion size: 6 by 6 pixels

Each source image produces four observations: one clean image and one image for each of the three corruption types. The prototype therefore contains 400 training and 400 test observations.

The original MNIST training and test splits remain separate. Automated checks verify the observation counts, image shape, finite values, pixel range, and equal representation of the four image conditions. Square occlusions are positioned over foreground pixels so that the intended corruption is visible.
