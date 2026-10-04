from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from datasets.galaxy_mnist import load_galaxy_mnist_catalog


CLASS_NAMES = {
    0: "smooth_round",
    1: "smooth_cigar",
    2: "edge_on_disk",
    3: "unbarred_spiral",
}


def main():
    catalog = load_galaxy_mnist_catalog(
        root=Path("data/GalaxyMNIST"),
        train=True,
    )

    samples_per_class = 3
    figure, axes = plt.subplots(
        nrows=4,
        ncols=samples_per_class,
        figsize=(8, 10),
    )

    for label, class_name in CLASS_NAMES.items():
        class_rows = catalog[catalog["label"] == label].sample(
            n=samples_per_class,
            random_state=1,
        )

        for column, (_, row) in enumerate(class_rows.iterrows()):
            image_path = Path(row["file_loc"])

            with Image.open(image_path) as image:
                axes[label, column].imshow(image.convert("RGB"))

            axes[label, column].axis("off")

            if column == 0:
                axes[label, column].set_title(
                    f"Class {label}: {class_name}",
                    loc="left",
                    fontsize=11,
                )

    figure.suptitle(
        "GalaxyMNIST morphology classes",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = Path(
        "reports/project-update/figures/galaxy_mnist_examples.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Figure saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()