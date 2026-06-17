"""
Loader for IBM Debater — Argument Quality Ranking dataset.

HuggingFace: ibm/argument_quality_ranking_30k
~30k argument–topic pairs with human quality scores.

We map this to the unified schema:
  - topic → Debate with a single claim (the motion)
  - stance_WA = 1  → premise      (argument supports the topic)
  - stance_WA = -1 → counter_claim (argument opposes the topic)
  - WA < 0.3       → unknown      (very low quality / unclear)
  - WA (weighted average quality) stored in metadata
"""

from typing import List, Dict

from data.schema import Argument, Debate

_UNKNOWN_QUALITY_THRESHOLD = 0.3


def _arg_type(stance_wa: int, quality: float) -> str:
    if quality < _UNKNOWN_QUALITY_THRESHOLD:
        return "unknown"
    return "premise" if stance_wa == 1 else "counter_claim"


def load_ibm(split: str = "train") -> List[Debate]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    ds = load_dataset(
        "ibm/argument_quality_ranking_30k",
        "argument_quality_ranking",
        split=split,
    )

    # Group arguments by topic so each topic becomes one Debate
    topics: Dict[str, List[dict]] = {}
    for row in ds:
        topics.setdefault(row["topic"], []).append(row)

    debates = []
    for topic, rows in topics.items():
        topic_id = topic.replace(" ", "_")[:64]
        claim = Argument(
            id=f"{topic_id}__claim",
            text=topic,
            arg_type="claim",
        )
        arguments = [
            Argument(
                id=f"{topic_id}__{i}",
                text=row["argument"],
                arg_type=_arg_type(row["stance_WA"], row["WA"]),
                parent_id=claim.id,
                metadata={
                    "quality_score": row["WA"],
                    "stance_wa": row["stance_WA"],
                },
            )
            for i, row in enumerate(rows)
        ]
        debates.append(
            Debate(
                id=topic_id,
                title=topic,
                source="ibm",
                arguments=[claim] + arguments,
            )
        )
    return debates
