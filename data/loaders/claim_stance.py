"""
Loader for IBM Claim Stance dataset.

HuggingFace: ibm/claim_stance
~2,400 claim-topic pairs where each claim is labeled PRO or CON
against a debate motion (e.g. "violent video games should be banned").

We map this to the unified schema:
  - topic → Debate with a single claim (the motion)
  - each row's claim text → Argument of type 'claim' or 'counter_claim'
    based on whether its stance agrees or opposes the topic sentiment
"""

from typing import List, Dict

from data.schema import Argument, Debate


def load_claim_stance(split: str = "train") -> List[Debate]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    # dataset only has 'train' and 'test' splits
    if split == "validation":
        split = "test"

    ds = load_dataset("ibm/claim_stance", "claim_stance", split=split)

    topics: Dict[str, List[dict]] = {}
    for row in ds:
        topics.setdefault(row["topicText"], []).append(row)

    debates = []
    for topic_text, rows in topics.items():
        topic_id = rows[0]["topicId"]
        motion = Argument(
            id=f"topic_{topic_id}",
            text=topic_text,
            arg_type="claim",
            metadata={"target": rows[0].get("topicTarget"), "sentiment": rows[0].get("topicSentiment")},
        )
        claims = [
            Argument(
                id=f"claim_{row['claims.claimId']}",
                text=row["claims.claimCorrectedText"],
                arg_type="claim" if row["claims.stance"] == "PRO" else "counter_claim",
                parent_id=motion.id,
                metadata={
                    "stance": row["claims.stance"],
                    "sentiment": row.get("claims.claimSentiment"),
                    "target": row.get("claims.claimTarget.text"),
                },
            )
            for row in rows
        ]
        debates.append(
            Debate(
                id=f"topic_{topic_id}",
                title=topic_text,
                source="ibm_claim_stance",
                arguments=[motion] + claims,
            )
        )
    return debates
