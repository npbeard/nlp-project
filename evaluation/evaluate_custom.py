from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src import LABEL2ID, predict


LABELS = list(LABEL2ID.keys())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {
                "example_id",
                "parent_text",
                "current_text",
                "gold_label",
            } - set(row)
            if missing:
                raise ValueError(
                    f"{path}:{line_no} missing fields: {sorted(missing)}"
                )
            if row["gold_label"] not in LABEL2ID:
                raise ValueError(
                    f"{path}:{line_no} invalid label: {row['gold_label']}"
                )
            rows.append(row)
    return rows


def evaluate(
    dataset_path: Path,
    checkpoint_dir: str,
    output_csv: Path,
) -> dict:
    rows = load_jsonl(dataset_path)
    y_true = []
    y_pred = []
    predictions = []

    for row in rows:
        pred = predict(
            text=row["current_text"],
            parent_text=row.get("parent_text", ""),
            checkpoint_dir=checkpoint_dir,
        )
        gold = row["gold_label"]
        y_true.append(gold)
        y_pred.append(pred)
        predictions.append({
            **row,
            "pred_label": pred,
            "correct": gold == pred,
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for prediction in predictions:
        for key in prediction:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "n_examples": len(rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro"),
        "classification_report": report,
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": matrix.tolist(),
        },
        "predictions_csv": str(output_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the argument role classifier on a custom JSONL set."
    )
    parser.add_argument(
        "--dataset",
        default="evaluation/custom_argument_eval.jsonl",
        type=Path,
    )
    parser.add_argument("--checkpoint", default="models/best")
    parser.add_argument(
        "--output-csv",
        default="evaluation/custom_argument_eval_predictions.csv",
        type=Path,
    )
    parser.add_argument(
        "--output-json",
        default="evaluation/custom_argument_eval_metrics.json",
        type=Path,
    )
    args = parser.parse_args()

    metrics = evaluate(args.dataset, args.checkpoint, args.output_csv)
    args.output_json.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Examples: {metrics['n_examples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            [r["gold_label"] for r in load_jsonl(args.dataset)],
            [
                r["pred_label"]
                for r in csv.DictReader(
                    args.output_csv.open(encoding="utf-8")
                )
            ],
            labels=LABELS,
            zero_division=0,
        )
    )
    print(f"Saved predictions to {args.output_csv}")
    print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
