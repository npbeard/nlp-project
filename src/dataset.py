from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from typing import List

from data.schema import Debate

LABEL2ID = {"claim": 0, "counter_claim": 1, "premise": 2, "unknown": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


class ArgumentDataset(Dataset):
    """Converts a list of Debate objects into tokenized (parent, argument) pairs.

    Each example feeds the parent comment as text_a and the reply as text_b so
    the model sees relational context — critical for distinguishing counter_claims
    from plain claims. Root arguments (no parent) are encoded as single sequences.
    """

    def __init__(
        self,
        debates: List[Debate],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._extract(debates)

    def _extract(self, debates: List[Debate]) -> list:
        examples = []
        for debate in debates:
            arg_map = {a.id: a for a in debate.arguments}
            for arg in debate.arguments:
                parent = arg_map.get(arg.parent_id) if arg.parent_id else None
                examples.append({
                    "text_a": parent.text if parent else "",
                    "text_b": arg.text,
                    "label": LABEL2ID[arg.arg_type],
                })
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        if ex["text_a"]:
            enc = self.tokenizer(
                ex["text_a"],
                ex["text_b"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            enc = self.tokenizer(
                ex["text_b"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(ex["label"], dtype=torch.long),
        }
