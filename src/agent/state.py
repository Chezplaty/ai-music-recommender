"""Data passed between the agent's steps."""

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from preferences import ParsedRequest
from recommender import Song, UserProfile
from rag.retrieval import RetrievedSong


@dataclass
class Strategy:
    """How the agent will build the playlist. The planner creates it; the AI critic revises it."""
    search_text: str
    n_candidates: int
    exclude_artists: List[str]
    max_per_artist: int
    profile: UserProfile                # working preferences; the AI critic may adjust them
    genre_targeted: bool = False        # also run one filtered search per requested genre
    banned_song_ids: List[int] = field(default_factory=list)  # songs the AI critic removed
    steps: List[str] = field(default_factory=list)  # human-readable plan, for the user


@dataclass
class Fact:
    """One piece of evidence about a song, computed from real data (catalog, scorer, retrieval)."""
    id: str                    # e.g. "energy", "genre", "search_match"
    text: str                  # e.g. "its energy of 0.48 is very close to your target of 0.45"
    supports: Optional[bool]   # True = a reason to pick it, False = a trade-off, None = neutral


@dataclass
class PlaylistItem:
    song: Song
    score: float                 # from the original weighted scorer
    reasons: List[str]           # the scorer's per-feature explanation
    similarity: Optional[float]  # retrieval match with the search text (None if not retrieved)
    confidence: str              # "high" / "medium" / "low", relative to the best possible score
    evidence: List[Fact] = field(default_factory=list)  # filled in by the explain step
    explanation: str = ""                                # why this song, in plain words
    cited: List[str] = field(default_factory=list)       # Fact ids the explanation is based on


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
class Revision:
    """A change of plan for the next attempt (from the AI critic, or the rule fallback)."""
    strategy: Strategy
    changes: List[str]  # human-readable diff, e.g. "energy 0.65 -> 0.55"
    source: str         # "ai" or "rules"


@dataclass
class AIReview:
    """The AI critic's verdict on one attempt."""
    mode: str                    # "critique" (rule checks passed) or "revise" (rule checks failed)
    approved: bool
    issues: List[str]
    reasoning: str
    revision: Optional[Revision] = None  # set when not approved
    error: str = ""                      # set when Gemini was unreachable


@dataclass
class Attempt:
    number: int
    strategy: Strategy
    playlist: List[PlaylistItem]
    evaluation: Evaluation
    review: Optional[AIReview] = None

    @property
    def accepted(self) -> bool:
        """Rule checks passed AND the AI critic approved (or couldn't be reached)."""
        if not self.evaluation.passed:
            return False
        return self.review is None or self.review.approved or bool(self.review.error)


@dataclass
class AgentResult:
    request: ParsedRequest
    playlist: List[PlaylistItem]
    evaluation: Evaluation   # of the playlist that was returned
    attempts: List[Attempt]  # every attempt, including revisions
    notes: List[str]         # limitations to tell the user about
    trace: List[Dict[str, str]]
    summary: str = ""        # one-paragraph description of the whole playlist
    chosen_attempt: int = 0  # number of the attempt whose playlist was returned


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
    final_strategy: Strategy  # strategy of the returned attempt (for explanations)
