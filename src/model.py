from __future__ import annotations

from transformers import RobertaForSequenceClassification

from .dataset import ID2LABEL, LABEL2ID, NUM_LABELS

MODEL_NAME = "roberta-base"


def build_model() -> RobertaForSequenceClassification:
    return RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
