import csv
import re
from collections import Counter
from pathlib import Path


ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports/final-report/llm-judge"
)

FIELDS = [
    "pair_id",
    "original_digit",
    "transformed_digit",
    "identity_preservation",
    "visible_transformation",
    "recognisability_B",
    "comment",
]


def main():
    records = []

    for start in range(1, 31, 5):
        end = start + 4
        path = ROOT / f"response_P{start:02d}_P{end:02d}.txt"
        batch = []

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not re.match(r"^P\d{2}\s+", line):
                continue

            values = line.split(maxsplit=6)
            if len(values) != 7:
                raise ValueError(f"Incomplete row in {path.name}: {line}")

            record = dict(zip(FIELDS, values))

            for field in ("original_digit", "transformed_digit"):
                if record[field] not in set("0123456789") | {"uncertain"}:
                    raise ValueError(f"Invalid {field}: {record}")

            if record["identity_preservation"] not in {
                "preserved", "ambiguous", "changed"
            }:
                raise ValueError(f"Invalid identity judgement: {record}")

            if record["visible_transformation"] not in {
                "rotation", "gaussian_noise", "square_occlusion",
                "none_visible", "uncertain",
            }:
                raise ValueError(f"Invalid transformation: {record}")

            if record["recognisability_B"] not in {"0", "1", "2"}:
                raise ValueError(f"Invalid recognisability: {record}")

            record["response_file"] = path.name
            batch.append(record)

        expected = [f"P{i:02d}" for i in range(start, end + 1)]
        if sorted(row["pair_id"] for row in batch) != expected:
            raise ValueError(f"Missing or duplicate pairs in {path.name}")

        records.extend(batch)

    with (ROOT / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        reference_rows = list(csv.DictReader(file))

    references = {row["pair_id"]: row for row in reference_rows}
    expected_ids = {row["pair_id"] for row in records}
    if len(reference_rows) != 30 or set(references) != expected_ids:
        raise ValueError("Manifest must contain exactly the same 30 pairs.")

    for row in records:
        reference = references[row["pair_id"]]
        row["source_index"] = reference["source_index"]
        row["reference_digit"] = reference["digit_label"]
        row["reference_corruption"] = reference["corruption"]
        row["original_matches_reference"] = (
            row["original_digit"] == reference["digit_label"]
        )
        row["transformed_matches_reference"] = (
            row["transformed_digit"] == reference["digit_label"]
        )
        row["transformation_matches_reference"] = (
            row["visible_transformation"] == reference["corruption"]
        )

    # Write derived results; preserve the original response files.
    output = ROOT / "judge_results.csv"
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    metrics = [
        ("Original digit matches reference", "original_matches_reference"),
        ("Transformed digit matches reference",
         "transformed_matches_reference"),
        ("Transformation matches reference",
         "transformation_matches_reference"),
    ]

    print(f"Pairs: {len(records)}")
    print(f"Source images: {len({r['source_index'] for r in records})}")

    for label, field in metrics:
        count = sum(row[field] for row in records)
        print(f"{label}: {count}/30 ({count / 30:.1%})")

    print("Identity judgements:", dict(Counter(
        row["identity_preservation"] for row in records
    )))
    print("Recognisability scores:", dict(Counter(
        row["recognisability_B"] for row in records
    )))

    print("\nPairs with a reference disagreement or non-preserved judgement:")
    for row in records:
        if (
            not all(row[field] for _, field in metrics)
            or row["identity_preservation"] != "preserved"
        ):
            print(
                row["pair_id"],
                f"reference_digit={row['reference_digit']}",
                f"original={row['original_digit']}",
                f"transformed={row['transformed_digit']}",
                f"reference_corruption={row['reference_corruption']}",
                f"judged_corruption={row['visible_transformation']}",
                f"identity={row['identity_preservation']}",
            )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()