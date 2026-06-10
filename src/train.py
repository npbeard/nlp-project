from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from data import load_ibm, clean_debates
from .dataset import ArgumentDataset, NUM_LABELS
from .model import MODEL_NAME, build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _class_weights(dataset: ArgumentDataset) -> torch.Tensor:
    """Inverse-frequency weights so rare labels (counter_claim, unknown) aren't ignored."""
    counts = Counter(ex["label"] for ex in dataset.examples)
    total = sum(counts.values())
    weights = torch.ones(NUM_LABELS)
    for label_id, count in counts.items():
        weights[label_id] = total / (NUM_LABELS * count)
    return weights


def evaluate(model, loader) -> tuple[float, float, float]:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            total_loss += loss_fn(logits, labels).item()
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return (
        total_loss / len(loader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="macro", zero_division=0),
    )


def train(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256,
    val_split: float = 0.15,
    output_dir: str = "models",
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading data...")
    debates = clean_debates(load_ibm("train"))
    print(f"  IBM Debater: {len(debates)} debates")

    try:
        from data import load_cmv
        cmv = clean_debates(load_cmv("train"))
        debates += cmv
        print(f"  CMV: {len(cmv)} debates  (total: {len(debates)})")
    except Exception:
        print("  CMV not found — training on IBM only")

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    dataset = ArgumentDataset(debates, tokenizer, max_length)
    print(f"  {len(dataset)} total examples")

    val_size = int(len(dataset) * val_split)
    train_set, val_set = random_split(
        dataset,
        [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model().to(DEVICE)
    weights = _class_weights(dataset).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = 0.0
    best_ckpt = os.path.join(output_dir, "best")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        val_loss, val_acc, val_f1 = evaluate(model, val_loader)
        print(
            f"Epoch {epoch}:  "
            f"train_loss={total_loss / len(train_loader):.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(best_ckpt)
            tokenizer.save_pretrained(best_ckpt)
            print(f"  → Saved best checkpoint (F1={best_f1:.4f}) to {best_ckpt}/")

    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_length": max_length,
                "best_val_f1": best_f1,
            },
            f,
            indent=2,
        )

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    return best_ckpt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train argument classifier. Run as: python -m src.train")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--output-dir", type=str, default="models")
    args = p.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        output_dir=args.output_dir,
    )
