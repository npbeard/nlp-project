"""
Text cleaning utilities shared across all loaders.

Applied before tokenization — keeps text usable for both
classical NLP pipelines and transformer tokenizers.
"""

import re
from typing import List

from data.schema import Argument, Debate

# Reddit-specific noise
_REDDIT_QUOTE = re.compile(r"^>.*$", re.MULTILINE)
_URL = re.compile(r"https?://\S+|www\.\S+")
_SUBREDDIT_MENTION = re.compile(r"r/\w+")
_USER_MENTION = re.compile(r"u/\w+")
_EDIT_NOTE = re.compile(r"\*?edit\*?:.*", re.IGNORECASE | re.DOTALL)
_WHITESPACE = re.compile(r"\s+")

# Deleted/removed placeholder strings
_DELETED = {"[deleted]", "[removed]", ""}


def clean_text(text: str) -> str:
    text = _REDDIT_QUOTE.sub("", text)
    text = _URL.sub("", text)
    text = _SUBREDDIT_MENTION.sub("", text)
    text = _USER_MENTION.sub("", text)
    text = _EDIT_NOTE.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def is_valid(text: str, min_tokens: int = 5) -> bool:
    """Return False for deleted posts or suspiciously short text."""
    if text in _DELETED:
        return False
    return len(text.split()) >= min_tokens


def clean_debate(debate: Debate, min_tokens: int = 5) -> Debate:
    """Return a new Debate with cleaned argument texts, dropping invalid ones."""
    cleaned_args = []
    for arg in debate.arguments:
        text = clean_text(arg.text)
        if is_valid(text, min_tokens):
            cleaned_args.append(
                Argument(
                    id=arg.id,
                    text=text,
                    arg_type=arg.arg_type,
                    parent_id=arg.parent_id,
                    author=arg.author,
                    score=arg.score,
                    metadata=arg.metadata,
                )
            )
    return Debate(
        id=debate.id,
        title=debate.title,
        source=debate.source,
        arguments=cleaned_args,
        metadata=debate.metadata,
    )


def clean_debates(debates: List[Debate], min_tokens: int = 5) -> List[Debate]:
    cleaned = [clean_debate(d, min_tokens) for d in debates]
    # Drop debates that lost their root claim during cleaning
    return [d for d in cleaned if d.root() is not None]
