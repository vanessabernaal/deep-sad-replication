from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    input_path = Path(
        "results/experiments/galaxy_experiment_summary.csv"
    )
    output_path = Path(
        "reports/project-update/figures/galaxy_pilot_auc.png"
    )

    summary = pd.read_csv(input_path)
    summary["labelled_anomaly_percent"] = (
        summary["ratio_known_outlier"] * 100
    )
    summary["test_auc_percent"] = summary["test_auc"] * 100

    grouped = (
        summary.groupby("labelled_anomaly_percent")[
            "test_auc_percent"
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.bar(
        grouped["labelled_anomaly_percent"].astype(str),
        grouped["mean"],
        yerr=grouped["std"],
        capsize=7,
        color=["#7f8c8d", "#4c78a8"],
        width=0.6,
    )

    for position, mean_value in enumerate(grouped["mean"]):
        axis.text(
            position,
            mean_value + 1.2,
            f"{mean_value:.2f}%",
            ha="center",
            fontweight="bold",
        )

    axis.set_xlabel("Labelled anomaly ratio (%)")
    axis.set_ylabel("Mean test ROC-AUC (%)")
    axis.set_title(
        "GalaxyMNIST pilot\n"
        "5 training epochs; mean ± SD across 3 seeds"
    )
    axis.set_ylim(50, 80)
    axis.grid(axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Figure saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()