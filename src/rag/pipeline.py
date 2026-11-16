"""
Retrieve-then-rank: the bridge between the RAG layer and the original recommender.

    ParsedRequest --(search text)--> ChromaDB --(candidate songs)--> score_song() --> playlist

Retrieval narrows 300 songs to a candidate pool that semantically fits the request
(genre style, sound, mood words, artist/album references). The existing weighted
scorer then ranks only those candidates, so the final order still comes from the
transparent formula rather than from an LLM or a black-box similarity.

With retriever=None the same function runs the original system (score the whole
catalog), which makes old-vs-RAG comparisons a one-flag change.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from preferences import ParsedRequest
from recommender import Recommender, Song
from rag.retrieval import MusicRetriever, RetrievedSong, is_excluded

MIN_CANDIDATES = 30
CANDIDATES_PER_SONG = 3  # pool size = max(MIN_CANDIDATES, 3 * songs requested)


@dataclass
class PlaylistResult:
    results: List[Tuple[Song, float, List[str]]]   # (song, score, reasons), best first
    mode: str                                       # "rag" or "baseline"
    search_text: str = ""
    retrieved: List[RetrievedSong] = field(default_factory=list)

    @property
    def similarity_by_id(self) -> Dict[int, float]:
        return {r.song.id: r.similarity for r in self.retrieved}


def build_search_text(request: ParsedRequest, query: str = "") -> str:
    """What we embed for the search: the LLM's sound description, plus artist references.

    The raw request ("I'm studying for finals...") is mostly about the activity, which
    song documents don't describe, so the sonic rewrite is preferred when available.
    """
    parts = [request.search_description or query]
    if request.reference_artists:
        parts.append("Similar to: " + ", ".join(request.reference_artists))
    return "\n".join(p for p in parts if p).strip()


def recommend(request: ParsedRequest, catalog: List[Song], retriever: Optional[MusicRetriever] = None,
              query: str = "", n_candidates: Optional[int] = None) -> PlaylistResult:
    k = request.num_songs

    if retriever is None:
        pool = [s for s in catalog if not is_excluded(s, request.exclude_artists)]
        return PlaylistResult(results=Recommender(pool).recommend_with_scores(request.profile, k=k), mode="baseline")

    search_text = build_search_text(request, query)
    n = n_candidates or max(MIN_CANDIDATES, CANDIDATES_PER_SONG * k)
    retrieved = retriever.search(search_text, n_results=n, exclude_artists=request.exclude_artists)
    candidates = [r.song for r in retrieved]
    return PlaylistResult(
        results=Recommender(candidates).recommend_with_scores(request.profile, k=k),
        mode="rag",
        search_text=search_text,
        retrieved=retrieved,
    )
