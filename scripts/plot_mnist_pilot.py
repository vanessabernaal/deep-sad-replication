from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    input_path = Path(
        "results/experiments/mnist_experiment_summary.csv"
    )
    output_path = Path(
        "reports/project-update/figures/mnist_pilot_auc.png"
    )

    summary = pd.read_csv(input_path)
    summary["labelled_anomaly_percent"] = (
        summary["ratio_known_outlier"] * 100
    )
    summary["test_auc_percent"] = summary["test_auc"] * 100

    grouped = (
        summary.groupby(
            ["normal_class", "labelled_anomaly_percent"]
        )["test_auc_percent"]
        .agg(["mean", "std"])
        .reset_index()
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    for normal_class in sorted(grouped["normal_class"].unique()):
        class_results = grouped[
            grouped["normal_class"] == normal_class
        ]

        axis.errorbar(
            class_results["labelled_anomaly_percent"],
            class_results["mean"],
            yerr=class_results["std"],
            marker="o",
            linewidth=2,
            capsize=5,
            label=f"Normal digit {normal_class}",
        )

    axis.set_xlabel("Labelled anomaly ratio (%)")
    axis.set_ylabel("Mean test ROC-AUC (%)")
    axis.set_title(
    "MNIST pilot: effect of labelled anomalies\n"
    "5 training epochs; mean ± SD across 3 seeds"
    )
    axis.set_xticks([0, 1, 5])
    axis.set_ylim(80, 100)
    axis.grid(alpha=0.3)
    axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Figure saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()