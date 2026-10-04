from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting


CORRUPTION_CLASSES = {
    1: "rotation",
    2: "gaussian_noise",
    3: "square_occlusion",
}


class ConstructedMNIST_Dataset(TorchvisionDataset):
    """Controlled-corruption MNIST dataset for Deep SAD."""

    def __init__(
        self,
        root: str,
        normal_class: int = 0,
        known_outlier_class: int = 1,
        n_known_outlier_classes: int = 0,
        ratio_known_normal: float = 0.0,
        ratio_known_outlier: float = 0.0,
        ratio_pollution: float = 0.0,
    ):
        super().__init__(root)

        if normal_class != 0:
            raise ValueError(
                "Constructed MNIST uses normal_class=0 for clean images."
            )

        if known_outlier_class not in CORRUPTION_CLASSES:
            raise ValueError(
                "known_outlier_class must be 1, 2, or 3 for "
                "rotation, gaussian noise, or square occlusion."
            )

        if n_known_outlier_classes not in (0, 1):
            raise ValueError(
                "Constructed MNIST supports zero or one known "
                "corruption class per experiment."
            )

        self.n_classes = 2
        self.normal_classes = (0,)
        self.outlier_classes = (1,)
        self.known_outlier_classes = (
            (1,) if n_known_outlier_classes == 1 else ()
        )
        self.corruption_name = CORRUPTION_CLASSES[
            known_outlier_class
        ]

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
        )

        idx, _, semi_targets = create_semisupervised_setting(
            np.asarray(train_set.targets),
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
        )


class ConstructedMNISTTensorDataset(Dataset):
    """Tensor dataset returning the four fields required by Deep SAD."""

    def __init__(self, data, corruption_name):
        selected_indices = [
            index
            for index, metadata in enumerate(data["metadata"])
            if metadata["corruption"] in ("clean", corruption_name)
        ]

        self.images = data["images"][selected_indices].float()
        self.digit_labels = data["digit_labels"][
            selected_indices
        ].long()
        self.metadata = [
            data["metadata"][index]
            for index in selected_indices
        ]

        self.targets = [
            0 if metadata["corruption"] == "clean" else 1
            for metadata in self.metadata
        ]

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