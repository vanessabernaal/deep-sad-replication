"""Prepare a fixed, stratified sample for manual visual quality review."""

import csv
import hashlib
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/constructed_mnist/train.pt"
OUTPUT_ROOT = PROJECT_ROOT / "reports/final-report/llm-judge"
SEED = 42
CORRUPTIONS = [
    "rotation",
    "gaussian_noise",
    "square_occlusion",
]


def main():
    # Avoid overwriting a sample that may already have been evaluated.
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"{OUTPUT_ROOT} already exists. Inspect it before rerunning."
        )

    data = torch.load(DATA_PATH, map_location="cpu")
    images = data["images"]
    labels = data["digit_labels"]
    metadata = data["metadata"]

    lookup = {}
    candidates = {digit: [] for digit in range(10)}

    for index, row in enumerate(metadata):
        assert row["split"] == "train"
        digit = int(labels[index])
        assert digit == int(row["digit_label"])

        key = (int(row["source_index"]), row["corruption"])
        assert key not in lookup
        lookup[key] = index

        if row["corruption"] == "clean":
            candidates[digit].append(key[0])

    rng = random.Random(SEED)
    selected = []

    for digit in range(10):
        source_index = rng.choice(sorted(candidates[digit]))
        for corruption in CORRUPTIONS:
            clean_index = lookup[(source_index, "clean")]
            transformed_index = lookup[(source_index, corruption)]
            assert int(labels[transformed_index]) == digit
            selected.append(
                (digit, source_index, corruption,
                 clean_index, transformed_index)
            )

    # Hide the digit-based ordering from the evaluator.
    rng.shuffle(selected)

    pair_root = OUTPUT_ROOT / "pairs"
    pair_root.mkdir(parents=True)
    manifest = []

    for number, item in enumerate(selected, start=1):
        digit, source_index, corruption, clean_index, transformed_index = item
        pair_id = f"P{number:02d}"
        filename = f"{pair_id}.png"

        fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.8))
        for ax, index, title in zip(
            axes,
            [clean_index, transformed_index],
            ["A: Original", "B: Transformed"],
        ):
            ax.imshow(
                images[index].squeeze().numpy(),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(pair_id)
        fig.tight_layout()
        fig.savefig(pair_root / filename, dpi=150)
        plt.close(fig)

        manifest.append({
            "pair_id": pair_id,
            "image_file": f"pairs/{filename}",
            "split": "train",
            "source_index": source_index,
            "digit_label": digit,
            "corruption": corruption,
            "severity": metadata[transformed_index]["severity"],
            "parameters": metadata[transformed_index]["parameters"],
        })

    with (OUTPUT_ROOT / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    protocol = {
        "sampling_seed": SEED,
        "source_split": "train",
        "source_images": 10,
        "pairs": 30,
        "sampling": "One randomly selected source per digit.",
        "pair_order": "Shuffled after selecting source images.",
        "source_file_sha256": hashlib.sha256(
            DATA_PATH.read_bytes()
        ).hexdigest(),
        "purpose": "Post-hoc exploratory visual quality review.",
        "selection_uses_deep_sad_scores": False,
        "evaluation_completed": False,
    }
    (OUTPUT_ROOT / "sample_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Prepared 30 pairs from 10 source images.")
    print(f"Images: {pair_root}")
    print("Manifest contains reference labels: do not upload it to the judge.")
    print(f"Protocol: {OUTPUT_ROOT / 'sample_protocol.json'}")


if __name__ == "__main__":
    main()