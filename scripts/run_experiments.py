import argparse
import csv
import json
import subprocess
import sys
from itertools import product
from pathlib import Path


EXPERIMENT_SUITES = {
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
        "known_outlier_classes": [1, 2, 3],
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
        choices=["mnist", "galaxy", "constructed", "all"],
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
        return ["mnist", "galaxy", "constructed"]
    return [suite_name]


def build_experiments(suite_name):
    suite = EXPERIMENT_SUITES[suite_name]

    known_outlier_classes = suite.get("known_outlier_classes")

    if known_outlier_classes is None:
        known_outlier_classes = [suite["known_outlier_class"]]

    for normal_class, known_outlier_class, ratio, seed in product(
        suite["normal_classes"],
        known_outlier_classes,
        suite["labelled_anomaly_ratios"],
        suite["seeds"],
    ):
        yield {
            "suite": suite_name,
            "dataset": suite["dataset"],
            "network": suite["network"],
            "normal_class": normal_class,
            "known_outlier_class": known_outlier_class,
            "ratio_known_outlier": ratio,
            "n_known_outlier_classes": 0 if ratio == 0.0 else 1,
            "seed": seed,
        }


def experiment_directory(results_root, experiment):
    directory = results_root / experiment["suite"]

    if experiment["suite"] == "constructed":
        corruption_names = {
            1: "rotation",
            2: "gaussian-noise",
            3: "square-occlusion",
        }
        corruption_name = corruption_names[
            experiment["known_outlier_class"]
        ]
        directory = directory / f"corruption-{corruption_name}"

    return (
        directory
        / f"normal-{experiment['normal_class']}"
        / f"ratio-{ratio_name(experiment['ratio_known_outlier'])}"
        / f"seed-{experiment['seed']}"
    )


def build_command(args, experiment, output_directory):
    return [
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


def main():
    args = parse_args()
    records = []

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
            results_path = output_directory / "results.json"

            if results_path.exists() and not args.force:
                status = "skipped_complete"
                print(
                    f"[{run_number}/{len(experiments)}] "
                    f"Skipping completed run: {output_directory}"
                )
            elif args.dry_run:
                status = "dry_run"
                command = build_command(
                    args,
                    experiment,
                    output_directory,
                )
                print(
                    f"[{run_number}/{len(experiments)}] "
                    + " ".join(command)
                )
            else:
                status = "completed"
                output_directory.mkdir(parents=True, exist_ok=True)

                command = build_command(
                    args,
                    experiment,
                    output_directory,
                )

                print(
                    f"\n[{run_number}/{len(experiments)}] "
                    f"Running: {output_directory}"
                )
                subprocess.run(command, check=True)

            results = read_result(output_directory, "results.json")
            ae_results = read_result(
                output_directory,
                "ae_results.json",
            )

            records.append(
                {
                    **experiment,
                    "n_epochs": args.n_epochs,
                    "ae_n_epochs": args.ae_n_epochs,
                    "status": status,
                    "test_auc": results.get("test_auc"),
                    "train_time": results.get("train_time"),
                    "test_time": results.get("test_time"),
                    "ae_test_auc": ae_results.get("test_auc"),
                    "ae_train_time": ae_results.get("train_time"),
                    "ae_test_time": ae_results.get("test_time"),
                    "output_directory": str(output_directory),
                }
            )

    write_summary(
        args.results_root,
        records,
        args.suite,
    )


if __name__ == "__main__":
    main()