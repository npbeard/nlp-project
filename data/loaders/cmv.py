"""
Loader for the Change My View (CMV) dataset.

Source: Tan et al. 2016 "Winning Arguments" — r/changemyview threads
where the OP awards a delta (∆) to a reply that changed their view.

Expected raw file: data/raw/cmv/train.jsonl (and test.jsonl)
Each line is a JSON object:
  {
    "id": "<reddit_post_id>",
    "title": "CMV: ...",
    "selftext": "<OP body>",
    "op_author": "<username>",
    "comments": [
      {
        "id": "<comment_id>",
        "body": "<text>",
        "author": "<username>",
        "score": <int>,
        "parent_id": "<post_id or comment_id>",
        "delta": <bool>   # True if OP awarded a delta to this comment
      }, ...
    ]
  }

Download: https://chenhaot.com/data/cmv/cmv.tar.bz2
"""

import json
from pathlib import Path
from typing import List

from data.schema import Argument, Debate

_CMV_RAW = Path("data/raw/cmv")


def _comment_to_arg_type(comment: dict) -> str:
    """
    Heuristic arg_type assignment:
    - delta-awarded comments are counter_claims (they changed the OP's mind)
    - top-level comments (parent == post) are claims
    - nested replies are premises
    """
    if comment.get("delta"):
        return "counter_claim"
    if str(comment.get("parent_id", "")).startswith("t3_"):  # t3_ = link/post prefix
        return "claim"
    return "premise"


def load_cmv(split: str = "train") -> List[Debate]:
    path = _CMV_RAW / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"CMV raw file not found at {path}.\n"
            "Download from https://chenhaot.com/data/cmv/cmv.tar.bz2 "
            "and extract into data/raw/cmv/"
        )

    debates = []
    with open(path) as f:
        for line in f:
            post = json.loads(line)
            args = [
                Argument(
                    id=post["id"],
                    text=post["selftext"],
                    arg_type="claim",
                    author=post.get("op_author"),
                    metadata={"title": post["title"]},
                )
            ]
            for c in post.get("comments", []):
                args.append(
                    Argument(
                        id=c["id"],
                        text=c["body"],
                        arg_type=_comment_to_arg_type(c),
                        parent_id=c.get("parent_id"),
                        author=c.get("author"),
                        score=c.get("score"),
                        metadata={"delta": c.get("delta", False)},
                    )
                )
            debates.append(
                Debate(
                    id=post["id"],
                    title=post["title"],
                    source="cmv",
                    arguments=args,
                )
            )
    return debates
