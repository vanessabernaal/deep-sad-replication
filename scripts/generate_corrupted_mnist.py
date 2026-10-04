"""Generate a reproducible controlled-corruption version of MNIST."""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import rotate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.mnist import MyMNIST  # noqa: E402


CORRUPTION_PARAMETERS = {
    1: {
        "rotation_degrees": 15.0,
        "noise_std": 0.15,
        "occlusion_size": 6,
    },
    2: {
        "rotation_degrees": 30.0,
        "noise_std": 0.30,
        "occlusion_size": 10,
    },
    3: {
        "rotation_degrees": 45.0,
        "noise_std": 0.45,
        "occlusion_size": 14,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate clean and corrupted MNIST observations."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing the original MyMNIST dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "constructed_mnist",
        help="Directory in which generated tensors and metadata are stored.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "project-update"
            / "figures"
            / "corrupted_mnist_examples.png"
        ),
        help="Path for the comparison figure.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of source images selected from each split.",
    )
    parser.add_argument(
        "--severity",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Corruption severity level.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed controlling sampling and transformations.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_stratified_indices(labels, max_samples, seed):
    """Select an approximately equal number of examples from each digit."""
    if max_samples <= 0:
        raise ValueError("--max-samples must be greater than zero.")

    if max_samples >= len(labels):
        return torch.arange(len(labels))

    generator = torch.Generator().manual_seed(seed)
    base_per_class = max_samples // 10
    remainder = max_samples % 10
    selected = []

    for digit in range(10):
        digit_indices = torch.where(labels == digit)[0]
        permutation = torch.randperm(
            len(digit_indices),
            generator=generator,
        )
        number_required = base_per_class + int(digit < remainder)
        selected.extend(
            digit_indices[permutation[:number_required]].tolist()
        )

    selected = torch.tensor(selected, dtype=torch.long)
    final_order = torch.randperm(
        len(selected),
        generator=generator,
    )
    return selected[final_order]


def apply_rotation(image, max_degrees, generator):
    sign = (
        -1.0
        if torch.rand(1, generator=generator).item() < 0.5
        else 1.0
    )
    angle = sign * max_degrees

    corrupted = rotate(image, angle=angle)
    parameters = {"angle_degrees": angle}

    return corrupted, parameters


def apply_gaussian_noise(image, noise_std, generator):
    noise = (
        torch.randn(
            image.shape,
            generator=generator,
        )
        * noise_std
    )
    corrupted = torch.clamp(image + noise, 0.0, 1.0)
    parameters = {"noise_std": noise_std}

    return corrupted, parameters


def apply_square_occlusion(image, square_size, generator):
    """Place a black square over a randomly selected foreground pixel."""
    height, width = image.shape[-2:]

    foreground = torch.nonzero(
        image[0] > 0.25,
        as_tuple=False,
    )

    if len(foreground) == 0:
        centre_y = height // 2
        centre_x = width // 2
    else:
        selected_position = int(
            torch.randint(
                0,
                len(foreground),
                (1,),
                generator=generator,
            ).item()
        )
        centre_y = int(foreground[selected_position, 0])
        centre_x = int(foreground[selected_position, 1])

    top = max(
        0,
        min(
            centre_y - square_size // 2,
            height - square_size,
        ),
    )
    left = max(
        0,
        min(
            centre_x - square_size // 2,
            width - square_size,
        ),
    )

    corrupted = image.clone()
    corrupted[
        :,
        top : top + square_size,
        left : left + square_size,
    ] = 0.0

    parameters = {
        "square_size": square_size,
        "top": top,
        "left": left,
        "foreground_threshold": 0.25,
    }

    return corrupted, parameters


def generate_split(dataset, split_name, indices, severity, seed):
    """Create clean, rotated, noisy, and occluded copies."""
    settings = CORRUPTION_PARAMETERS[severity]
    generator = torch.Generator().manual_seed(seed)

    generated_images = []
    digit_labels = []
    anomaly_targets = []
    metadata = []

    for source_index in indices.tolist():
        image = (
            dataset.data[source_index]
            .float()
            .unsqueeze(0)
            / 255.0
        )
        digit_label = int(dataset.targets[source_index])

        rotated_image, rotation_parameters = apply_rotation(
            image,
            settings["rotation_degrees"],
            generator,
        )
        noisy_image, noise_parameters = apply_gaussian_noise(
            image,
            settings["noise_std"],
            generator,
        )
        occluded_image, occlusion_parameters = apply_square_occlusion(
            image,
            settings["occlusion_size"],
            generator,
        )

        variants = [
            ("clean", image.clone(), {}),
            (
                "rotation",
                rotated_image,
                rotation_parameters,
            ),
            (
                "gaussian_noise",
                noisy_image,
                noise_parameters,
            ),
            (
                "square_occlusion",
                occluded_image,
                occlusion_parameters,
            ),
        ]

        for corruption, transformed_image, parameters in variants:
            generated_images.append(transformed_image)
            digit_labels.append(digit_label)
            anomaly_targets.append(
                int(corruption != "clean")
            )

            metadata.append(
                {
                    "split": split_name,
                    "source_index": source_index,
                    "digit_label": digit_label,
                    "corruption": corruption,
                    "severity": (
                        0 if corruption == "clean" else severity
                    ),
                    "parameters": json.dumps(
                        parameters,
                        sort_keys=True,
                    ),
                    "seed": seed,
                }
            )

    payload = {
        "images": torch.stack(generated_images),
        "digit_labels": torch.tensor(
            digit_labels,
            dtype=torch.long,
        ),
        "anomaly_targets": torch.tensor(
            anomaly_targets,
            dtype=torch.long,
        ),
        "metadata": metadata,
    }

    return payload


def validate_payload(payload, expected_sources):
    """Run basic checks before saving generated data."""
    images = payload["images"]
    metadata = payload["metadata"]

    assert images.shape == (
        expected_sources * 4,
        1,
        28,
        28,
    )
    assert len(payload["digit_labels"]) == expected_sources * 4
    assert len(payload["anomaly_targets"]) == expected_sources * 4
    assert len(metadata) == expected_sources * 4

    assert torch.isfinite(images).all()
    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0

    corruption_counts = Counter(
        row["corruption"] for row in metadata
    )
    expected_corruptions = {
        "clean",
        "rotation",
        "gaussian_noise",
        "square_occlusion",
    }

    assert set(corruption_counts) == expected_corruptions
    assert all(
        count == expected_sources
        for count in corruption_counts.values()
    )

    return corruption_counts


def write_metadata_csv(
    train_metadata,
    test_metadata,
    output_path,
):
    rows = train_metadata + test_metadata
    fieldnames = [
        "split",
        "source_index",
        "digit_label",
        "corruption",
        "severity",
        "parameters",
        "seed",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def create_comparison_figure(
    payload,
    output_path,
    examples=5,
):
    """Create a grid with one source image per row."""
    images = payload["images"]
    labels = payload["digit_labels"]

    titles = [
        "Clean",
        "Rotation",
        "Gaussian noise",
        "Square occlusion",
    ]

    rows = min(examples, len(images) // 4)

    figure, axes = plt.subplots(
        rows,
        4,
        figsize=(8, 2 * rows),
    )

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(rows):
        for column in range(4):
            position = row * 4 + column

            axes[row, column].imshow(
                images[position, 0],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row, column].axis("off")

            if row == 0:
                axes[row, column].set_title(
                    titles[column]
                )

            if column == 0:
                axes[row, column].set_ylabel(
                    f"Digit {int(labels[position])}",
                    rotation=90,
                )

    figure.suptitle(
        "Controlled-corruption MNIST examples"
    )
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    args = parse_args()
    set_seed(args.seed)

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.figure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset = MyMNIST(
        root=str(args.data_root),
        train=True,
        download=True,
    )
    test_dataset = MyMNIST(
        root=str(args.data_root),
        train=False,
        download=True,
    )

    train_indices = select_stratified_indices(
        train_dataset.targets,
        args.max_samples,
        args.seed,
    )
    test_indices = select_stratified_indices(
        test_dataset.targets,
        args.max_samples,
        args.seed + 1,
    )

    train_payload = generate_split(
        train_dataset,
        "train",
        train_indices,
        args.severity,
        args.seed,
    )
    test_payload = generate_split(
        test_dataset,
        "test",
        test_indices,
        args.severity,
        args.seed + 1,
    )

    train_counts = validate_payload(
        train_payload,
        len(train_indices),
    )
    test_counts = validate_payload(
        test_payload,
        len(test_indices),
    )

    torch.save(
        train_payload,
        args.output_root / "train.pt",
    )
    torch.save(
        test_payload,
        args.output_root / "test.pt",
    )

    write_metadata_csv(
        train_payload["metadata"],
        test_payload["metadata"],
        args.output_root / "metadata.csv",
    )

    summary = {
        "seed": args.seed,
        "severity": args.severity,
        "corruption_parameters": (
            CORRUPTION_PARAMETERS[args.severity]
        ),
        "source_images_per_split": args.max_samples,
        "generated_images_per_split": (
            args.max_samples * 4
        ),
        "train_corruption_counts": dict(train_counts),
        "test_corruption_counts": dict(test_counts),
        "image_shape": [1, 28, 28],
        "value_range": [0.0, 1.0],
    }

    with (
        args.output_root / "summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    create_comparison_figure(
        test_payload,
        args.figure_path,
    )

    print("Constructed MNIST generation completed.")
    print(
        f"Train observations: "
        f"{len(train_payload['images'])}"
    )
    print(
        f"Test observations: "
        f"{len(test_payload['images'])}"
    )
    print(f"Data saved to: {args.output_root}")
    print(f"Figure saved to: {args.figure_path}")
    print("All validation checks passed.")


if __name__ == "__main__":
    main()