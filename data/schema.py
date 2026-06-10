from dataclasses import dataclass, field
from typing import List, Optional, Dict

ARG_TYPES = {"claim", "counter_claim", "premise", "unknown"}


@dataclass
class Argument:
    id: str
    text: str
    arg_type: str  # one of ARG_TYPES
    parent_id: Optional[str] = None
    author: Optional[str] = None
    score: Optional[int] = None  # Reddit upvotes/downvotes
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.arg_type not in ARG_TYPES:
            raise ValueError(f"arg_type must be one of {ARG_TYPES}, got '{self.arg_type}'")


@dataclass
class Debate:
    id: str
    title: str
    source: str  # 'cmv', 'ibm', 'reddit'
    arguments: List[Argument] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def root(self) -> Optional[Argument]:
        """The top-level claim (no parent)."""
        roots = [a for a in self.arguments if a.parent_id is None]
        return roots[0] if roots else None

    def claims(self) -> List[Argument]:
        return [a for a in self.arguments if a.arg_type == "claim"]

    def counter_claims(self) -> List[Argument]:
        return [a for a in self.arguments if a.arg_type == "counter_claim"]

    def premises(self) -> List[Argument]:
        return [a for a in self.arguments if a.arg_type == "premise"]

    def replies_to(self, argument_id: str) -> List[Argument]:
        return [a for a in self.arguments if a.parent_id == argument_id]

    def __len__(self) -> int:
        return len(self.arguments)
