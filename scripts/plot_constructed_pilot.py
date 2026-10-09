from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = (
    PROJECT_ROOT / "results/experiments-50ep-ae100"
)
OUTPUT_ROOT = (
    PROJECT_ROOT / "reports/project-update/figures"
)

CONDITION_NAMES = {
    "clean": "Clean",
    "rotation": "Rotation",
    "gaussian_noise": "Gaussian noise",
    "square_occlusion": "Square occlusion",
}


def main():
    # Combine the clean control and corruption experiments.
    results = pd.concat(
        [
            pd.read_csv(
                RESULTS_ROOT
                / "constructed_clean_experiment_summary.csv"
            ),
            pd.read_csv(
                RESULTS_ROOT
                / "constructed_experiment_summary.csv"
            ),
        ],
        ignore_index=True,
    )

    # Check that all 24 runs belong to the intended configuration.
    assert len(results) == 24, "Expected 24 runs."
    assert results["status"].eq("completed").all()
    assert results["n_epochs"].eq(50).all()
    assert results["ae_n_epochs"].eq(100).all()
    assert results["normal_class"].eq(0).all()
    assert results["known_outlier_class"].eq(1).all()
    assert set(results["corruption"]) == set(CONDITION_NAMES)
    assert set(results["ratio_known_outlier"]) == {0.0, 0.05}

    for _, group in results.groupby(
        ["corruption", "ratio_known_outlier"]
    ):
        assert len(group) == 3
        assert set(group["seed"]) == {1, 2, 3}

    results["test_auc_percent"] = results["test_auc"] * 100

    summary = (
        results.groupby(
            ["corruption", "ratio_known_outlier"]
        )["test_auc_percent"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_path = (
        OUTPUT_ROOT / "constructed_robustness_auc_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    order = list(CONDITION_NAMES)
    positions = np.arange(len(order))
    offsets = [-0.12, 0.12]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Points and error bars show means and sample standard deviations.
    for ratio, offset, colour, label in zip(
        [0.0, 0.05],
        offsets,
        ["#4C78A8", "#F58518"],
        ["0% labelled anomalies", "5% labelled anomalies"],
    ):
        condition = (
            summary[
                summary["ratio_known_outlier"] == ratio
            ]
            .set_index("corruption")
            .reindex(order)
        )

        ax.errorbar(
            positions + offset,
            condition["mean"],
            yerr=condition["std"],
            fmt="o",
            markersize=7,
            capsize=5,
            color=colour,
            label=label,
        )

    ax.axhline(
        50,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Chance level",
    )
    ax.set_title(
        "Deep SAD on clean and corrupted MNIST\n"
        "50 training epochs; 100 pretraining epochs"
    )
    ax.set_ylabel("Test ROC-AUC (%)")
    ax.set_xlabel("Image condition")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [CONDITION_NAMES[name] for name in order]
    )
    ax.set_ylim(45, 101)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=1,
        loc="lower left",
    )

    fig.text(
        0.5,
        0.015,
        "Mean ± sample SD across 3 seeds. "
        "Each model is trained and tested in the same image condition.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    figure_path = (
        OUTPUT_ROOT / "constructed_robustness_auc.png"
    )
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"\nTable saved to: {summary_path}")
    print(f"Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()