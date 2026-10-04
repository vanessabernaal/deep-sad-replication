from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
)

from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting


GALAXY_MNIST_RESOURCES = {
    "train_catalog": (
        "https://dl.dropboxusercontent.com/s/6xlym0ney5q9aec14vvy1/"
        "galaxy_mnist_train_catalog.parquet"
        "?rlkey=fc5knqscwk2h6r4z5d3156vzr&dl=0",
        "galaxy_mnist_train_catalog.parquet",
        "cd22b21d165802f4bc1adf997424aec2",
    ),
    "test_catalog": (
        "https://dl.dropboxusercontent.com/s/9assxy247i1nq8wjy7o0a/"
        "galaxy_mnist_test_catalog.parquet"
        "?rlkey=y1ns24xw0tcyntodi61rv14fu&dl=0",
        "galaxy_mnist_test_catalog.parquet",
        "c5cca8d8afb6fb0d59baeb310a35d594",
    ),
    "images": (
        "https://dl.dropboxusercontent.com/s/lj89307kjx5plme9f66js/"
        "galaxy_mnist_images.tar.gz"
        "?rlkey=gvlwx2rl3hqpo3gplb9zlqrz0",
        "galaxy_mnist_images.tar.gz",
        "1c6cb0447f2f7ed676c3363ee194ced9",
    ),
}


def prepare_galaxy_mnist(root: Path) -> None:
    """Download and extract the official GalaxyMNIST files when required."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    for key in ("train_catalog", "test_catalog"):
        url, filename, md5 = GALAXY_MNIST_RESOURCES[key]
        destination = root / filename

        if not check_integrity(str(destination), md5):
            download_url(
                url=url,
                root=str(root),
                filename=filename,
                md5=md5,
            )

    images_url, archive_name, archive_md5 = GALAXY_MNIST_RESOURCES["images"]
    archive_path = root / archive_name
    image_directory = root / "images"

    images_exist = (
        image_directory.exists()
        and any(image_directory.glob("*.jpg"))
    )

    if not images_exist:
        download_and_extract_archive(
            url=images_url,
            download_root=str(root),
            filename=archive_name,
            md5=archive_md5,
        )
    elif not check_integrity(str(archive_path), archive_md5):
        raise RuntimeError(
            "GalaxyMNIST images exist, but the downloaded archive failed "
            "its integrity check."
        )


def load_galaxy_mnist_catalog(root: Path, train: bool):
    """Load the official train or test catalogue and create image paths."""
    root = Path(root)
    prepare_galaxy_mnist(root)

    filename = (
        "galaxy_mnist_train_catalog.parquet"
        if train
        else "galaxy_mnist_test_catalog.parquet"
    )

    catalog = pd.read_parquet(root / filename).copy()
    catalog["file_loc"] = catalog["filename"].map(
        lambda filename_value: str(root / "images" / filename_value)
    )

    missing_images = [
        file_location
        for file_location in catalog["file_loc"]
        if not Path(file_location).exists()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} GalaxyMNIST images are missing."
        )

    return catalog


class GalaxyMNIST_Dataset(TorchvisionDataset):
    """GalaxyMNIST adaptation for Deep SAD."""

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

        all_classes = list(range(4))

        if normal_class not in all_classes:
            raise ValueError("normal_class must be one of 0, 1, 2, or 3.")

        self.n_classes = 2
        self.normal_classes = (normal_class,)
        self.outlier_classes = tuple(
            label for label in all_classes if label != normal_class
        )

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            if known_outlier_class not in self.outlier_classes:
                raise ValueError(
                    "known_outlier_class must differ from normal_class."
                )
            self.known_outlier_classes = (known_outlier_class,)
        else:
            self.known_outlier_classes = tuple(
                random.sample(
                    list(self.outlier_classes),
                    n_known_outlier_classes,
                )
            )

        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )

        target_transform = transforms.Lambda(
            lambda target: int(target in self.outlier_classes)
        )

        galaxy_root = Path(self.root) / "GalaxyMNIST"

        train_catalog = load_galaxy_mnist_catalog(
            root=galaxy_root,
            train=True,
        )

        train_set = GalaxyMNISTTorchDataset(
            catalog=train_catalog,
            transform=transform,
            target_transform=target_transform,
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

        test_catalog = load_galaxy_mnist_catalog(
            root=galaxy_root,
            train=False,
        )

        self.test_set = GalaxyMNISTTorchDataset(
            catalog=test_catalog,
            transform=transform,
            target_transform=target_transform,
        )


class GalaxyMNISTTorchDataset(Dataset):
    """PyTorch wrapper returning the four fields required by Deep SAD."""

    def __init__(self, catalog, transform=None, target_transform=None):
        self.catalog = catalog.reset_index(drop=True).copy()
        self.targets = self.catalog["label"].astype(int).tolist()
        self.semi_targets = torch.zeros(
            len(self.catalog),
            dtype=torch.int64,
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, index):
        row = self.catalog.iloc[index]
        image_path = Path(row["file_loc"])
        target = int(row["label"])
        semi_target = int(self.semi_targets[index])

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, semi_target, index