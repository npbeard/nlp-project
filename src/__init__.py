from __future__ import annotations

import importlib.metadata
from packaging.version import Version

_tv = Version(importlib.metadata.version("transformers"))
if _tv >= Version("5.0.0"):
    raise RuntimeError(
        f"transformers {_tv} is installed but this project requires <5.0.0. "
        "Run: pip install transformers==4.57.6"
    )

from data.schema import Argument, Debate

LABEL2ID = {"claim": 0, "counter_claim": 1, "premise": 2, "unknown": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

_model = None
_tokenizer = None
_loaded_checkpoint: str | None = None
_device: str | None = None


def _load(checkpoint_dir: str) -> None:
    global _model, _tokenizer, _loaded_checkpoint, _device
    if _loaded_checkpoint != checkpoint_dir:
        import torch
        from transformers import (
            RobertaForSequenceClassification,
            RobertaTokenizerFast,
        )
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint_dir)
        _model = RobertaForSequenceClassification.from_pretrained(
            checkpoint_dir
        )
        _model.eval()
        _model.to(_device)
        _loaded_checkpoint = checkpoint_dir


def predict(
    text: str,
    parent_text: str = "",
    checkpoint_dir: str = "models/best",
) -> str:
    """Classify a single argument text.

    Returns one of: 'claim', 'counter_claim', 'premise', 'unknown'.
    Pass parent_text when the comment is a reply.
    """
    import torch
    _load(checkpoint_dir)
    if parent_text:
        enc = _tokenizer(
            parent_text,
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length",
        )
    else:
        enc = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length",
        )
    enc = {k: v.to(_device) for k, v in enc.items()}
    with torch.no_grad():
        logits = _model(**enc).logits
    return ID2LABEL[logits.argmax(dim=-1).item()]


def predict_debate(
    debate: Debate,
    checkpoint_dir: str = "models/best",
) -> Debate:
    """Label every argument in a debate, return new Debate with predictions.

    Main entry point for Person 3 (eval) and Person 4 (failure analysis).
    Preserves structure and parent_id links; only arg_type is replaced.
    """
    _load(checkpoint_dir)
    arg_map = {a.id: a for a in debate.arguments}
    labeled = []
    for arg in debate.arguments:
        parent = arg_map.get(arg.parent_id) if arg.parent_id else None
        labeled.append(Argument(
            id=arg.id,
            text=arg.text,
            arg_type=predict(
                arg.text,
                parent.text if parent else "",
                checkpoint_dir,
            ),
            parent_id=arg.parent_id,
            author=arg.author,
            score=arg.score,
            metadata=arg.metadata,
        ))
    return Debate(
        id=debate.id,
        title=debate.title,
        source=debate.source,
        arguments=labeled,
        metadata=debate.metadata,
    )


__all__ = ["predict", "predict_debate", "LABEL2ID", "ID2LABEL"]
