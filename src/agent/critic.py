"""
The AI critic: Gemini judges each attempt and decides how to revise the plan.

Two modes, chosen by the workflow:
  critique  The guardrail rules passed. Gemini judges what rules can't: does this
            playlist actually fit the ORIGINAL request ("late-night drive",
            "not too distracting", "the vibe of Blonde")? It approves, or rejects
            and writes the next plan.
  revise    The guardrail rules failed. The verdict is already "no"; Gemini
            diagnoses why and writes the next plan.

What Gemini can change (its "levers"): the search text, genres, moods, energy/
valence/acousticness targets, which songs to remove, the candidate pool size,
and whether to search inside each genre. It can NOT add songs by name: every
song still comes from the ChromaDB catalog and is ranked by the original scorer.

apply_review() is the guardrail on the critic itself: unknown genres/moods are
dropped, numbers are clamped, and only songs in the current playlist can be
removed. A failed rule check can never be overridden by an AI approval.
"""

import logging
from dataclasses import dataclass, replace
from typing import List, Optional

from pydantic import BaseModel, Field

from preferences import MOOD_DEFINITIONS, LLMError, ParsedRequest, generate_structured
from recommender import UserProfile
from agent.evaluator import MAX_CANDIDATES, MIN_CANDIDATES
from agent.state import AIReview, Evaluation, PlaylistItem, Revision, Strategy

logger = logging.getLogger(__name__)

MAX_SEARCH_TEXT_CHARS = 500
MAX_ISSUES = 8


class CriticError(LLMError):
    """The critic call failed (network, overload, bad output)."""


class ReviewResponse(BaseModel):
    """Schema Gemini fills in. Field descriptions are sent to the model."""
    approved: bool = Field(description="True only if the playlist is a good fit for the ORIGINAL request as it is.")
    issues: List[str] = Field(description="Specific problems, citing song ids, e.g. '[117] is a worship song, not a "
                              "late-night drive song'. Empty if approved.")
    reasoning: str = Field(description="One or two sentences explaining the verdict and the revision strategy.")
    next_search_text: str = Field(description="Sound description to search for next attempt. Describe the SOUND "
                                  "(genre style, mood, energy, instrumentation, production), not the activity. "
                                  "Copy the current one if it should not change.")
    next_genres: List[str] = Field(description="Genres for next attempt, from the allowed list. Copy current if unchanged.")
    next_moods: List[str] = Field(description="Moods for next attempt, from the allowed list. Copy current if unchanged.")
    next_target_energy: float = Field(description="0-1. Copy current if unchanged.")
    next_target_valence: float = Field(description="0-1. Copy current if unchanged.")
    next_target_acousticness: float = Field(description="0-1. Copy current if unchanged.")
    remove_song_ids: List[int] = Field(description="Ids of songs in this playlist that clearly do not fit and must "
                                       "not be picked again. Empty if none.")
    next_n_candidates: int = Field(description=f"How many songs to retrieve next time ({MIN_CANDIDATES}-"
                                   f"{MAX_CANDIDATES}). Larger finds more options but strays further from the search text.")
    search_within_genres: bool = Field(description="True to also run a separate search inside each genre in "
                                       "next_genres (use when results drift away from the requested genres).")


@dataclass
class ReviewContext:
    mode: str                 # "critique" or "revise"
    query: str                # the user's original words
    request: ParsedRequest
    strategy: Strategy
    playlist: List[PlaylistItem]
    evaluation: Evaluation
    attempt_number: int
    revisions_left: int
    vocab: dict               # allowed genres / moods


def build_system_prompt(vocab: dict) -> str:
    moods = "\n".join(f"  - {m}: {MOOD_DEFINITIONS.get(m, '')}" for m in vocab["moods"])
    return f"""You are the critic and planner inside a music playlist agent.

Each attempt, the agent searches a fixed song catalog (semantic search over song
descriptions), ranks the results with a weighted scorer, and applies guardrail
rules. You see the result and decide what happens next.

Your job:
1. Judge the playlist against the user's ORIGINAL request: activity, vibe,
   reference artists/albums, constraints like "not distracting" or "new artists".
   Use your own knowledge of the songs and artists. The catalog's genre and mood
   labels come from automatic heuristics and can be wrong.
2. If it is a good fit, approve it. Do not reject for perfection: each revision
   costs the user time, and the catalog is small (300 songs, 12 per genre).
3. If it is not a good fit, write the plan for the next attempt using the levers
   in the response schema. You cannot name new songs; you steer the search.

Rules for the next plan:
- Keep the user's explicit wishes. Only change genres/moods/targets when the
  current interpretation misses the request, and say why in `reasoning`.
- Only use these genres: {", ".join(vocab["genres"])}
- Only use these moods:
{moods}
- Only remove songs by ids shown in the playlist.
"""


def build_review_prompt(ctx: ReviewContext) -> str:
    s, p = ctx.strategy, ctx.strategy.profile
    rows = "\n".join(
        f"[{i.song.id}] {i.song.title} - {i.song.artist} | album: {i.song.album or '?'} | {i.song.genre} | "
        f"{i.song.mood} | energy {i.song.energy:.2f} | valence {i.song.valence:.2f} | "
        f"acousticness {i.song.acousticness:.2f} | {i.song.tempo_bpm:.0f} BPM | score {i.score:.2f}"
        + (f" | match {i.similarity:.2f}" if i.similarity is not None else "")
        for i in ctx.playlist
    ) or "(empty)"
    checks = "\n".join(f"{'PASS' if c.passed else 'FAIL'} {c.name}: {c.detail}" for c in ctx.evaluation.checks)

    if ctx.mode == "critique":
        task = ("The guardrail checks all PASSED. Decide whether this playlist truly fits the ORIGINAL REQUEST. "
                "Approve it, or reject it and write the next plan.")
    else:
        task = ("The guardrail checks FAILED (see FAIL lines). This attempt cannot be approved: set approved=false, "
                "diagnose why it failed, and write a next plan that fixes it.")
    if ctx.revisions_left == 0:
        task += " This is the LAST attempt: give an honest verdict; your plan will not be used."

    return f"""ORIGINAL REQUEST: "{ctx.query}"
Songs requested: {ctx.request.num_songs}. Reference artists: {ctx.request.reference_artists or 'none'}. \
Wants new artists: {ctx.request.discover_new_artists}.

CURRENT PLAN (attempt {ctx.attempt_number}, revisions left: {ctx.revisions_left})
search_text: {s.search_text}
genres: {p.genres}  moods: {p.moods}
target_energy: {p.target_energy:.2f}  target_valence: {p.target_valence:.2f}  target_acousticness: {p.target_acousticness:.2f}
candidate pool: {s.n_candidates}  search_within_genres: {s.genre_targeted}
excluded artists: {s.exclude_artists or 'none'}  max songs per artist: {s.max_per_artist}
removed songs so far: {s.banned_song_ids or 'none'}

PLAYLIST
{rows}

GUARDRAIL CHECKS
{checks}

TASK: {task}"""


class GeminiReviewer:
    """Default reviewer: one Gemini call per review. Raises CriticError on failure."""

    def __init__(self, client=None, model: Optional[str] = None):
        self.client = client
        self.model = model

    def __call__(self, ctx: ReviewContext) -> ReviewResponse:
        return generate_structured(build_review_prompt(ctx), build_system_prompt(ctx.vocab), ReviewResponse,
                                   client=self.client, model=self.model, error_cls=CriticError)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _valid_subset(proposed: List[str], allowed: set, current: List[str]) -> List[str]:
    """Allowed values from `proposed`, deduped. If none are allowed, keep `current`
    (an all-invalid answer must not silently become 'any genre')."""
    cleaned = []
    for x in proposed:
        x = x.strip().lower()
        if x in allowed and x not in cleaned:
            cleaned.append(x)
    if proposed and not cleaned:
        return list(current)
    return cleaned


def apply_review(ctx: ReviewContext, resp: ReviewResponse) -> AIReview:
    """Validate the critic's answer and turn it into an AIReview (with a Revision if rejected)."""
    approved = bool(resp.approved) and ctx.mode == "critique"  # rules failed -> AI can't approve
    issues = [i.strip() for i in resp.issues if i and i.strip()][:MAX_ISSUES]
    review = AIReview(mode=ctx.mode, approved=approved, issues=issues, reasoning=resp.reasoning.strip())
    if approved:
        return review

    s, p = ctx.strategy, ctx.strategy.profile
    genres = _valid_subset(resp.next_genres, set(ctx.vocab["genres"]), p.genres)
    moods = _valid_subset(resp.next_moods, set(ctx.vocab["moods"]), p.moods)
    profile = UserProfile(genres=genres, moods=moods,
                          target_energy=_clamp(resp.next_target_energy),
                          target_valence=_clamp(resp.next_target_valence),
                          target_acousticness=_clamp(resp.next_target_acousticness))
    search_text = resp.next_search_text.strip()[:MAX_SEARCH_TEXT_CHARS] or s.search_text
    playlist_ids = {i.song.id for i in ctx.playlist}
    removed = [sid for sid in dict.fromkeys(resp.remove_song_ids) if sid in playlist_ids]
    n = max(MIN_CANDIDATES, min(MAX_CANDIDATES, int(resp.next_n_candidates)))

    new = replace(s, search_text=search_text, profile=profile, n_candidates=n,
                  genre_targeted=bool(resp.search_within_genres),
                  banned_song_ids=s.banned_song_ids + [x for x in removed if x not in s.banned_song_ids])

    changes = []
    if search_text != s.search_text:
        changes.append(f'search text -> "{search_text}"')
    for name in ("genres", "moods"):
        if set(getattr(profile, name)) != set(getattr(p, name)):  # a reorder is not a change
            changes.append(f"{name} {getattr(p, name)} -> {getattr(profile, name)}")
    for name in ("target_energy", "target_valence", "target_acousticness"):
        old, now = getattr(p, name), getattr(profile, name)
        if abs(old - now) >= 0.01:
            changes.append(f"{name.replace('target_', '')} {old:.2f} -> {now:.2f}")
    if removed:
        titles = {i.song.id: i.song.title for i in ctx.playlist}
        changes.append("removed " + ", ".join(f"[{x}] {titles[x]}" for x in removed))
    if n != s.n_candidates:
        changes.append(f"candidate pool {s.n_candidates} -> {n}")
    if new.genre_targeted != s.genre_targeted:
        changes.append(f"search within each genre: {new.genre_targeted}")

    review.revision = Revision(strategy=new, changes=changes, source="ai")
    return review


def run_review(reviewer, ctx: ReviewContext) -> AIReview:
    """Call the reviewer; a failure becomes an AIReview with `error` set (the workflow falls back to rules)."""
    try:
        return apply_review(ctx, reviewer(ctx))
    except LLMError as e:
        logger.warning("AI critic unavailable: %s", e)
        return AIReview(mode=ctx.mode, approved=False, issues=[], reasoning="", error=str(e))
