from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = ["claim", "counter_claim", "premise", "unknown"]


def plot_confusion_matrix(metrics_path: Path, output_path: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    labels = metrics["confusion_matrix"]["labels"]
    matrix = np.array(metrics["confusion_matrix"]["matrix"])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)

    max_value = matrix.max() if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > max_value / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_label_distribution(predictions_path: Path, output_path: Path) -> None:
    with predictions_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    gold_counts = Counter(row["gold_label"] for row in rows)
    pred_counts = Counter(row["pred_label"] for row in rows)
    x = np.arange(len(LABELS))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        [gold_counts[label] for label in LABELS],
        width,
        label="Gold",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        [pred_counts[label] for label in LABELS],
        width,
        label="Predicted",
        color="#F58518",
    )
    ax.set_title("Gold vs Predicted Label Distribution")
    ax.set_ylabel("Number of examples")
    ax.set_xticks(x, LABELS, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(gold_counts.values()), max(pred_counts.values())) + 3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot custom evaluation results."
    )
    parser.add_argument(
        "--metrics",
        default="evaluation/custom_argument_eval_metrics.json",
        type=Path,
    )
    parser.add_argument(
        "--predictions",
        default="evaluation/custom_argument_eval_predictions.csv",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/figures",
        type=Path,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = args.output_dir / "confusion_matrix.png"
    distribution_path = args.output_dir / "label_distribution.png"

    plot_confusion_matrix(args.metrics, confusion_path)
    plot_label_distribution(args.predictions, distribution_path)

    print(f"Saved {confusion_path}")
    print(f"Saved {distribution_path}")


if __name__ == "__main__":
    main()
