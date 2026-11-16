"""Data passed between the agent's steps."""

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from preferences import ParsedRequest
from recommender import Song
from rag.retrieval import RetrievedSong


@dataclass
class Strategy:
    """How the agent will build the playlist. The planner creates it; the reviser changes it."""
    search_text: str
    n_candidates: int
    exclude_artists: List[str]
    max_per_artist: int
    genre_targeted: bool = False  # also run one filtered search per requested genre
    steps: List[str] = field(default_factory=list)  # human-readable plan, for the user


@dataclass
class PlaylistItem:
    song: Song
    score: float                 # from the original weighted scorer
    reasons: List[str]           # the scorer's per-feature explanation
    similarity: Optional[float]  # retrieval match with the search text (None if not retrieved)
    confidence: str              # "high" / "medium" / "low", relative to the best possible score


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    hard: bool = False  # hard = a guardrail (catalog, duplicates, exclusions); soft = quality


@dataclass
class Evaluation:
    checks: List[Check]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]


@dataclass
class Attempt:
    number: int
    strategy: Strategy
    playlist: List[PlaylistItem]
    evaluation: Evaluation


@dataclass
class AgentResult:
    request: ParsedRequest
    playlist: List[PlaylistItem]
    evaluation: Evaluation   # of the playlist that was returned
    attempts: List[Attempt]  # every attempt, including revisions
    notes: List[str]         # limitations to tell the user about
    trace: List[Dict[str, str]]


class AgentState(TypedDict, total=False):
    """LangGraph state. Each node returns only the keys it changes.

    `trace` and `attempts` use operator.add, so each node's entries are appended
    instead of overwriting earlier ones.
    """
    query: str
    request: ParsedRequest
    strategy: Strategy
    notes: List[str]
    retrieved: List[RetrievedSong]
    playlist: List[PlaylistItem]
    evaluation: Evaluation
    revisions: int
    gave_up: bool  # the reviser found no fix worth trying
    attempts: Annotated[List[Attempt], operator.add]
    trace: Annotated[List[Dict[str, Any]], operator.add]
    result: AgentResult
