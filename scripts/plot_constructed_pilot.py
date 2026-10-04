from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_PATH = Path(
    "results/experiments/constructed_experiment_summary.csv"
)
OUTPUT_PATH = Path(
    "reports/project-update/figures/"
    "constructed_mnist_pilot_auc.png"
)

CORRUPTION_NAMES = {
    1: "Rotation",
    2: "Gaussian noise",
    3: "Square occlusion",
}


def main():
    results = pd.read_csv(RESULTS_PATH)

    results["corruption"] = results[
        "known_outlier_class"
    ].map(CORRUPTION_NAMES)

    results["test_auc_percent"] = (
        results["test_auc"] * 100
    )

    summary = (
        results.groupby(
            ["corruption", "ratio_known_outlier"]
        )["test_auc_percent"]
        .agg(["mean", "std"])
        .reset_index()
    )

    corruption_order = [
        "Rotation",
        "Gaussian noise",
        "Square occlusion",
    ]
    ratios = [0.0, 0.05]
    ratio_labels = ["0% labelled anomalies", "5% labelled anomalies"]
    colours = ["#4C78A8", "#F58518"]

    x_positions = np.arange(len(corruption_order))
    bar_width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for index, (ratio, label, colour) in enumerate(
        zip(ratios, ratio_labels, colours)
    ):
        condition = (
            summary[
                summary["ratio_known_outlier"] == ratio
            ]
            .set_index("corruption")
            .reindex(corruption_order)
        )

        positions = (
            x_positions
            + (index - 0.5) * bar_width
        )

        bars = ax.bar(
            positions,
            condition["mean"],
            width=bar_width,
            yerr=condition["std"],
            capsize=5,
            label=label,
            color=colour,
            alpha=0.9,
        )

        for bar, value in zip(bars, condition["mean"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.axhline(
        50,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="Chance level",
    )

    ax.set_title(
        "Constructed MNIST pilot: Deep SAD detection performance\n"
        "5 training epochs; mean ± SD across 3 seeds"
    )
    ax.set_xlabel("Corruption type")
    ax.set_ylabel("Test ROC-AUC (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(corruption_order)
    ax.set_ylim(20, 75)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()

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

    print(f"Figure saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()