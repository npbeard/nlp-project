"""
Loader for IBM Debater — Argument Quality Ranking dataset.

HuggingFace: ibm/argument_quality_ranking_30k
~30k argument–topic pairs with human quality scores.

We map this to the unified schema:
  - topic → Debate with a single claim (the motion)
  - argument → Argument of type 'premise' attached to that claim
  - WA (weighted average quality) stored in metadata
"""

from typing import List, Dict

from data.schema import Argument, Debate


def load_ibm(split: str = "train") -> List[Debate]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    ds = load_dataset("ibm/argument_quality_ranking_30k", "argument_quality_ranking", split=split)

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
        premises = [
            Argument(
                id=f"{topic_id}__{i}",
                text=row["argument"],
                arg_type="premise",
                parent_id=claim.id,
                metadata={
                    "quality_score": row.get("WA"),
                    "stance": row.get("stance"),
                },
            )
            for i, row in enumerate(rows)
        ]
        debates.append(
            Debate(
                id=topic_id,
                title=topic,
                source="ibm",
                arguments=[claim] + premises,
            )
        )
    return debates
