from __future__ import annotations

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

from data.schema import Argument, Debate
from .dataset import ID2LABEL, LABEL2ID

_model = None
_tokenizer = None
_loaded_checkpoint: str | None = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load(checkpoint_dir: str) -> None:
    global _model, _tokenizer, _loaded_checkpoint
    if _loaded_checkpoint != checkpoint_dir:
        _tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint_dir)
        _model = RobertaForSequenceClassification.from_pretrained(checkpoint_dir)
        _model.eval()
        _model.to(_device)
        _loaded_checkpoint = checkpoint_dir


def predict(text: str, parent_text: str = "", checkpoint_dir: str = "models/best") -> str:
    """Classify a single argument text.

    Returns one of: 'claim', 'counter_claim', 'premise', 'unknown'.
    Pass parent_text when the comment is a reply — the model uses both together.
    """
    _load(checkpoint_dir)
    if parent_text:
        enc = _tokenizer(
            parent_text, text,
            return_tensors="pt", truncation=True, max_length=256, padding="max_length",
        )
    else:
        enc = _tokenizer(
            text,
            return_tensors="pt", truncation=True, max_length=256, padding="max_length",
        )
    enc = {k: v.to(_device) for k, v in enc.items()}
    with torch.no_grad():
        logits = _model(**enc).logits
    return ID2LABEL[logits.argmax(dim=-1).item()]


def predict_debate(debate: Debate, checkpoint_dir: str = "models/best") -> Debate:
    """Classify every argument in a debate and return a new Debate with predicted labels.

    This is the main entry point for Person 3 (eval) and Person 4 (failure analysis).
    The returned Debate has the same structure and parent_id links — just with
    arg_type replaced by the model's predictions.
    """
    _load(checkpoint_dir)
    arg_map = {a.id: a for a in debate.arguments}
    labeled = []
    for arg in debate.arguments:
        parent = arg_map.get(arg.parent_id) if arg.parent_id else None
        labeled.append(Argument(
            id=arg.id,
            text=arg.text,
            arg_type=predict(arg.text, parent.text if parent else "", checkpoint_dir),
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
