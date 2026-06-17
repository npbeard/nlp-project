"""
Loader for the ConvoKit Winning Arguments (CMV) corpus.

Source: Tan et al. 2016 — r/changemyview threads where OP awards
        a delta to a reply that changed their view.
Corpus: Cornell ConvoKit winning-args-corpus (3,051 threads, 293k utts)

Download once:
    python -c "from convokit import download; download('winning-args-corpus')"

Label mapping:
  - root utterance (OP post) → claim
  - meta['success'] == 1     → counter_claim (delta-awarded)
  - meta['success'] == 0     → premise (argument, no delta)
  - meta['success'] is None  → unknown (not in a pair)

max_unknown caps the unknown class so it doesn't overwhelm training.
"""

import json
import random
from pathlib import Path
from typing import List

from data.schema import Argument, Debate

_CORPUS_PATH = (
    Path.home() / ".convokit" / "saved-corpora" / "winning-args-corpus"
)


def load_convokit_cmv(
    split: str = "train",
    max_unknown: int = 15_000,
    seed: int = 42,
) -> List[Debate]:
    utterances_path = _CORPUS_PATH / "utterances.jsonl"
    conversations_path = _CORPUS_PATH / "conversations.json"

    if not utterances_path.exists():
        raise FileNotFoundError(
            f"ConvoKit corpus not found at {_CORPUS_PATH}.\n"
            "Run: python -c "
            "\"from convokit import download; "
            "download('winning-args-corpus')\""
        )

    with open(conversations_path, encoding="utf-8") as f:
        conv_meta = json.load(f)

    valid_ids = {
        conv_id
        for conv_id, data in conv_meta.items()
        if bool(data.get("meta", {}).get("train", True)) == (split == "train")
    }

    conv_utterances: dict = {}
    with open(utterances_path, encoding="utf-8") as f:
        for line in f:
            utt = json.loads(line)
            root = utt["root"]
            if root not in valid_ids:
                continue
            conv_utterances.setdefault(root, []).append(utt)

    rng = random.Random(seed)

    debates = []
    for conv_id, utterances in conv_utterances.items():
        meta = conv_meta.get(conv_id, {}).get("meta", {})
        title = meta.get("op-title", "")

        root_utt = next((u for u in utterances if u["id"] == conv_id), None)
        if root_utt is None:
            continue

        claim = Argument(
            id=root_utt["id"],
            text=str(root_utt.get("text") or meta.get("op-text-body", "")),
            arg_type="claim",
            author=root_utt.get("user"),
            metadata={"title": title},
        )

        structured, unknowns = [claim], []
        for utt in utterances:
            if utt["id"] == conv_id:
                continue
            text = str(utt.get("text") or "")
            if not text.strip():
                continue

            success = utt.get("meta", {}).get("success")
            if success == 1:
                arg_type = "counter_claim"
            elif success == 0:
                arg_type = "premise"
            else:
                arg_type = "unknown"

            arg = Argument(
                id=utt["id"],
                text=text,
                arg_type=arg_type,
                parent_id=utt.get("reply-to"),
                author=utt.get("user"),
                score=utt.get("meta", {}).get("score"),
                metadata={"success": success},
            )
            if arg_type == "unknown":
                unknowns.append(arg)
            else:
                structured.append(arg)

        debates.append(Debate(
            id=conv_id,
            title=title,
            source="cmv",
            arguments=structured,
            metadata={"unknowns": unknowns},
        ))

    # Collect all unknowns, sample down, distribute back
    all_unknowns = [
        u for d in debates for u in d.metadata.get("unknowns", [])
    ]
    sampled = set(
        u.id for u in rng.sample(
            all_unknowns, min(max_unknown, len(all_unknowns))
        )
    )
    for debate in debates:
        kept = [u for u in debate.metadata["unknowns"] if u.id in sampled]
        debate.arguments.extend(kept)
        del debate.metadata["unknowns"]

    return debates
