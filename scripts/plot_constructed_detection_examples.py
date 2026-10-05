import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.main import load_dataset


RESULTS_PATH = (
    PROJECT_ROOT
    / "results/experiments/constructed/"
    / "corruption-rotation/normal-0/"
    / "ratio-05pct/seed-1/results.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports/project-update/figures/"
    / "constructed_detection_examples.png"
)


def main():
    dataset = load_dataset(
        dataset_name="constructed_mnist",
        data_path=str(PROJECT_ROOT / "data"),
        normal_class=0,
        known_outlier_class=1,
        n_known_outlier_classes=1,
        ratio_known_normal=0.0,
        ratio_known_outlier=0.05,
        ratio_pollution=0.0,
    )

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    records = [
        {
            "index": int(index),
            "target": int(target),
            "score": float(score),
        }
        for index, target, score in results["test_scores"]
    ]

    anomalous = [
        record for record in records
        if record["target"] == 1
    ]
    normal = [
        record for record in records
        if record["target"] == 0
    ]

    detected_anomalies = sorted(
        anomalous,
        key=lambda record: record["score"],
        reverse=True,
    )[:4]

    missed_anomalies = sorted(
        anomalous,
        key=lambda record: record["score"],
    )[:4]

    incorrectly_flagged = sorted(
        normal,
        key=lambda record: record["score"],
        reverse=True,
    )[:4]

    groups = [
        (
            "Detected anomalies\n(highest anomaly scores)",
            detected_anomalies,
        ),
        (
            "Missed anomalies\n(lowest anomaly scores)",
            missed_anomalies,
        ),
        (
            "Incorrectly flagged clean\n(highest clean scores)",
            incorrectly_flagged,
        ),
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(9, 7),
    )

    for row, (group_name, group_records) in enumerate(groups):
        for column, record in enumerate(group_records):
            image, target, _, index = dataset.test_set[
                record["index"]
            ]
            metadata = dataset.test_set.metadata[index]

            axes[row, column].imshow(
                image.squeeze(0).numpy(),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row, column].set_title(
                f"Digit {metadata['digit_label']}\n"
                f"Score: {record['score']:.3f}",
                fontsize=9,
            )
            axes[row, column].axis("off")

    row_positions = [0.73, 0.45, 0.17]

    for position, (group_name, _) in zip(
        row_positions,
        groups,
    ):
        fig.text(
            0.035,
            position,
            group_name,
            ha="center",
            va="center",
            rotation=90,
            fontsize=10,
        )

    fig.suptitle(
        "Constructed MNIST qualitative error analysis\n"
        "Rotation, 5% labelled anomalies, seed 1",
        fontsize=14,
    )

    fig.tight_layout(rect=[0.10, 0, 1, 0.94])

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Figure saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()