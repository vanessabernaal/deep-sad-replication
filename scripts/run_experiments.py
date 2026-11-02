import argparse
import csv
import json
import subprocess
import sys
from itertools import product
from pathlib import Path


EXPERIMENT_SUITES = {
    "constructed_clean": {
        "dataset": "constructed_mnist",
        "network": "mnist_LeNet",
        "normal_classes": [0],
        "known_outlier_class": 1,
        "corruptions": ["clean"],
        "labelled_anomaly_ratios": [0.0, 0.05],
        "seeds": [1, 2, 3],
    },
    "mnist": {
        "dataset": "mnist",
        "network": "mnist_LeNet",
        "normal_classes": [0, 3],
        "known_outlier_class": 1,
        "labelled_anomaly_ratios": [0.0, 0.01, 0.05],
        "seeds": [1, 2, 3],
    },
    "galaxy": {
        "dataset": "galaxy_mnist",
        "network": "cifar10_LeNet",
        "normal_classes": [0],
        "known_outlier_class": 1,
        "labelled_anomaly_ratios": [0.0, 0.05],
        "seeds": [1, 2, 3],
    },

    "constructed": {
        "dataset": "constructed_mnist",
        "network": "mnist_LeNet",
        "normal_classes": [0],
        "known_outlier_class": 1,
        "corruptions": [
            "rotation",
            "gaussian_noise",
            "square_occlusion",
        ],
        "labelled_anomaly_ratios": [0.0, 0.05],
        "seeds": [1, 2, 3],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run reproducible Deep SAD experiment suites."
    )
    parser.add_argument(
        "--suite",
        choices=["mnist", "galaxy", "constructed", "constructed_clean", "all"],
        required=True,
        help="Experiment suite to execute.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        required=True,
        help="Number of Deep SAD training epochs.",
    )
    parser.add_argument(
        "--ae-n-epochs",
        type=int,
        required=True,
        help="Number of autoencoder pretraining epochs.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device. Default: cpu.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data"),
        help="Dataset directory. Default: data.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/experiments"),
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training and pretraining batch size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running experiments.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run experiments even if results.json already exists.",
    )
    return parser.parse_args()


def ratio_name(ratio):
    return f"{int(round(ratio * 100)):02d}pct"


def selected_suites(suite_name):
    if suite_name == "all":
        return [
            "mnist",
            "galaxy",
            "constructed",
            "constructed_clean",
        ]
    return [suite_name]


def build_experiments(suite_name):
    suite = EXPERIMENT_SUITES[suite_name]
    corruptions = suite.get("corruptions", [None])

    for normal_class, corruption, ratio, seed in product(
        suite["normal_classes"],
        corruptions,
        suite["labelled_anomaly_ratios"],
        suite["seeds"],
    ):
        yield {
            "suite": suite_name,
            "dataset": suite["dataset"],
            "network": suite["network"],
            "normal_class": normal_class,
            "known_outlier_class": suite[
                "known_outlier_class"
            ],
            "corruption": corruption,
            "ratio_known_outlier": ratio,
            "n_known_outlier_classes": (
                0 if ratio == 0.0 else 1
            ),
            "seed": seed,
        }


def experiment_directory(results_root, experiment):
    if experiment["suite"] == "constructed":
        directory = (
            results_root
            / "constructed-robustness"
            / (
                "corruption-"
                + experiment["corruption"].replace("_", "-")
            )
        )
    else:
        directory = (
            results_root
            / experiment["suite"]
        )

    return (
        directory
        / f"normal-{experiment['normal_class']}"
        / (
            "ratio-"
            + ratio_name(
                experiment["ratio_known_outlier"]
            )
        )
        / f"seed-{experiment['seed']}"
    )


def build_command(args, experiment, output_directory):
    command = [
        sys.executable,
        "src/main.py",
        experiment["dataset"],
        experiment["network"],
        str(output_directory),
        str(args.data_path),
        "--device",
        args.device,
        "--normal_class",
        str(experiment["normal_class"]),
        "--known_outlier_class",
        str(experiment["known_outlier_class"]),
        "--n_known_outlier_classes",
        str(experiment["n_known_outlier_classes"]),
        "--ratio_known_outlier",
        str(experiment["ratio_known_outlier"]),
        "--ratio_pollution",
        "0.0",
        "--seed",
        str(experiment["seed"]),
        "--n_epochs",
        str(args.n_epochs),
        "--ae_n_epochs",
        str(args.ae_n_epochs),
        "--batch_size",
        str(args.batch_size),
        "--ae_batch_size",
        str(args.batch_size),
    ]
    if experiment["corruption"] is not None:
        command.extend([
            "--corruption",
            experiment["corruption"],
        ])

    return command


def read_result(output_directory, filename):
    file_path = output_directory / filename

    if not file_path.exists():
        return {}

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_summary(results_root, records, suite_name):
    summary_path = (
        results_root
        / f"{suite_name}_experiment_summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "suite",
        "dataset",
        "network",
        "corruption",
        "normal_class",
        "known_outlier_class",
        "n_known_outlier_classes",
        "ratio_known_outlier",
        "seed",
        "n_epochs",
        "ae_n_epochs",
        "status",
        "test_auc",
        "train_time",
        "test_time",
        "ae_test_auc",
        "ae_train_time",
        "ae_test_time",
        "output_directory",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSummary saved to: {summary_path}")

def validate_saved_run(args, experiment, output_directory):
    """Check that saved results match the requested configuration."""
    config_path = output_directory / "config.json"

    if not config_path.exists():
        raise ValueError(
            f"Missing configuration: {config_path}. "
            "Use a new results directory."
        )

    config = read_result(output_directory, "config.json")

    expected = {
        "dataset_name": experiment["dataset"],
        "net_name": experiment["network"],
        "normal_class": experiment["normal_class"],
        "known_outlier_class": experiment["known_outlier_class"],
        "n_known_outlier_classes": experiment["n_known_outlier_classes"],
        "ratio_known_outlier": experiment["ratio_known_outlier"],
        "ratio_known_normal": 0.0,
        "ratio_pollution": 0.0,
        "seed": experiment["seed"],
        "n_epochs": args.n_epochs,
        "ae_n_epochs": args.ae_n_epochs,
        "batch_size": args.batch_size,
        "ae_batch_size": args.batch_size,
        "pretrain": True,
    }

    if experiment["corruption"] is not None:
        expected["corruption"] = experiment["corruption"]

    mismatches = [
        f"{key}: saved={config.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]

    if mismatches:
        raise ValueError(
            f"Configuration mismatch in {output_directory}:\n"
            + "\n".join(mismatches)
            + "\nUse a different --results-root to preserve this run."
        )

    results = read_result(output_directory, "results.json")
    ae_results = read_result(output_directory, "ae_results.json")

    if results.get("test_auc") is None:
        raise ValueError(
            f"Missing Deep SAD AUC in {output_directory}."
        )

    if ae_results.get("test_auc") is None:
        raise ValueError(
            f"Missing autoencoder AUC in {output_directory}."
        )

    return config, results, ae_results


def main():
    args = parse_args()
    records = []

    if min(args.n_epochs, args.ae_n_epochs, args.batch_size) <= 0:
        raise ValueError(
            "Epoch counts and batch size must be positive."
        )

    for suite_name in selected_suites(args.suite):
        experiments = list(build_experiments(suite_name))

        print(
            f"\nSuite '{suite_name}': "
            f"{len(experiments)} configured experiments"
        )

        for run_number, experiment in enumerate(experiments, start=1):
            output_directory = experiment_directory(
                args.results_root,
                experiment,
            )
            command = build_command(
                args,
                experiment,
                output_directory,
            )
            prefix = f"[{run_number}/{len(experiments)}]"

            # Dry runs only print commands; they never write summaries.
            if args.dry_run:
                print(f"{prefix} " + " ".join(command))
                continue

            results_path = output_directory / "results.json"

            if results_path.exists() and not args.force:
                config, results, ae_results = validate_saved_run(
                    args,
                    experiment,
                    output_directory,
                )
                status = "skipped_complete"
                print(
                    f"{prefix} Reusing verified run: "
                    f"{output_directory}"
                )
            else:
                output_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                print(
                    f"\n{prefix} Running: {output_directory}"
                )
                subprocess.run(command, check=True)

                config, results, ae_results = validate_saved_run(
                    args,
                    experiment,
                    output_directory,
                )
                status = "completed"

            records.append(
                {
                    **experiment,
                    "n_epochs": config["n_epochs"],
                    "ae_n_epochs": config["ae_n_epochs"],
                    "status": status,
                    "test_auc": results["test_auc"],
                    "train_time": results.get("train_time"),
                    "test_time": results.get("test_time"),
                    "ae_test_auc": ae_results["test_auc"],
                    "ae_train_time": ae_results.get("train_time"),
                    "ae_test_time": ae_results.get("test_time"),
                    "output_directory": str(output_directory),
                }
            )

    if args.dry_run:
        print("\nDry run completed. No files were written.")
        return

    write_summary(
        args.results_root,
        records,
        args.suite,
    )


if __name__ == "__main__":
    main()