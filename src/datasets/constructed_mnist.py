from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting


VALID_CORRUPTIONS = (
    "clean",
    "rotation",
    "gaussian_noise",
    "square_occlusion",
)


class ConstructedMNIST_Dataset(TorchvisionDataset):
    """MNIST evaluated under one controlled corruption condition."""

    def __init__(
        self,
        root: str,
        normal_class: int = 0,
        known_outlier_class: int = 1,
        n_known_outlier_classes: int = 0,
        ratio_known_normal: float = 0.0,
        ratio_known_outlier: float = 0.0,
        ratio_pollution: float = 0.0,
        corruption: str = "rotation",
    ):
        super().__init__(root)

        if normal_class not in range(10):
            raise ValueError(
                "normal_class must be an MNIST digit from 0 to 9."
            )

        if known_outlier_class not in range(10):
            raise ValueError(
                "known_outlier_class must be an MNIST digit from 0 to 9."
            )

        if known_outlier_class == normal_class:
            raise ValueError(
                "known_outlier_class must differ from normal_class."
            )

        if n_known_outlier_classes not in (0, 1):
            raise ValueError(
                "Constructed MNIST supports zero or one known "
                "anomaly class."
            )

        if corruption not in VALID_CORRUPTIONS:
            raise ValueError(
                f"corruption must be one of {VALID_CORRUPTIONS}."
            )

        self.n_classes = 2
        self.normal_classes = (normal_class,)
        self.outlier_classes = tuple(
            digit
            for digit in range(10)
            if digit != normal_class
        )
        self.known_outlier_classes = (
            (known_outlier_class,)
            if n_known_outlier_classes == 1
            else ()
        )
        self.corruption_name = corruption

        dataset_root = Path(self.root) / "constructed_mnist"
        train_path = dataset_root / "train.pt"
        test_path = dataset_root / "test.pt"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "Constructed MNIST files were not found. Run "
                "scripts/generate_corrupted_mnist.py first."
            )

        train_data = torch.load(
            train_path,
            map_location="cpu",
        )
        test_data = torch.load(
            test_path,
            map_location="cpu",
        )

        train_set = ConstructedMNISTTensorDataset(
            data=train_data,
            corruption_name=self.corruption_name,
            normal_class=normal_class,
        )

        idx, _, semi_targets = create_semisupervised_setting(
            train_set.digit_labels.cpu().numpy(),
            self.normal_classes,
            self.outlier_classes,
            self.known_outlier_classes,
            ratio_known_normal,
            ratio_known_outlier,
            ratio_pollution,
        )

        train_set.semi_targets[idx] = torch.tensor(
            semi_targets,
            dtype=torch.int64,
        )

        self.train_set = Subset(train_set, idx)

        self.test_set = ConstructedMNISTTensorDataset(
            data=test_data,
            corruption_name=self.corruption_name,
            normal_class=normal_class,
        )


class ConstructedMNISTTensorDataset(Dataset):
    """Tensor dataset returning the fields required by Deep SAD."""

    def __init__(
        self,
        data,
        corruption_name,
        normal_class,
    ):
        selected_indices = [
            index
            for index, metadata in enumerate(data["metadata"])
            if metadata["corruption"] == corruption_name
        ]

        if not selected_indices:
            raise ValueError(
                f"No observations found for {corruption_name}."
            )

        self.images = data["images"][
            selected_indices
        ].float()

        self.digit_labels = data["digit_labels"][
            selected_indices
        ].long()

        self.metadata = [
            data["metadata"][index]
            for index in selected_indices
        ]

        self.targets = (
            self.digit_labels != normal_class
        ).long()

        self.semi_targets = torch.zeros(
            len(self.targets),
            dtype=torch.int64,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self.images[index]
        target = int(self.targets[index])
        semi_target = int(self.semi_targets[index])

        return image, target, semi_target, index