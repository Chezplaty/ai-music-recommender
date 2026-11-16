"""
The critic: checks a playlist against the request, and decides how to revise if it falls short.

Hard checks are guardrails (the generator already enforces them; checking again
catches bugs). Soft checks measure quality and are what usually trigger a revision.
Thresholds are deliberately simple so every verdict can be explained in one line.
"""

from collections import Counter
from dataclasses import replace
from statistics import mean
from typing import List, Optional, Tuple

from preferences import ParsedRequest
from recommender import Song
from guardrails.validation import duplicate_songs, invalid_songs, primary_artist
from rag.retrieval import is_excluded
from agent.generator import max_possible_score
from agent.state import Check, Evaluation, PlaylistItem, Strategy

MIN_GENRE_SHARE = 0.6     # at least 60% of songs in a requested genre...
MAX_WEAK_SHARE = 0.3      # ...at most 30% "weak" picks...
WEAK_SCORE_RATIO = 0.70   # (weak = below 70% of the best possible score)
MAX_ENERGY_GAP = 0.15     # ...and average energy within 0.15 of the target
MIN_CANDIDATES = 10
MAX_CANDIDATES = 150


def available_songs(catalog: List[Song], strategy: Strategy) -> List[Song]:
    return [s for s in catalog if not is_excluded(s, strategy.exclude_artists)]


def evaluate(playlist: List[PlaylistItem], request: ParsedRequest, strategy: Strategy,
             catalog: List[Song]) -> Evaluation:
    songs = [item.song for item in playlist]
    profile = strategy.profile
    pool = available_songs(catalog, strategy)
    target = min(request.num_songs, len(pool))
    checks = []

    # --- hard guardrails ---
    bad = invalid_songs(songs, catalog)
    checks.append(Check("in_catalog", not bad, "all songs are real catalog entries" if not bad
                        else f"not in catalog: {[s.title for s in bad]}", hard=True))
    dupes = duplicate_songs(songs)
    checks.append(Check("no_duplicates", not dupes, "no duplicate songs" if not dupes
                        else f"duplicates: {[s.title for s in dupes]}", hard=True))
    excluded = [s for s in songs if is_excluded(s, strategy.exclude_artists)]
    checks.append(Check("excluded_artists", not excluded,
                        "no excluded artists" if not excluded else f"excluded artists present: {[s.artist for s in excluded]}",
                        hard=True))
    over = {a: n for a, n in Counter(primary_artist(s) for s in songs).items() if n > strategy.max_per_artist}
    checks.append(Check("artist_variety", not over, f"at most {strategy.max_per_artist} song(s) per artist"
                        if not over else f"too many songs by {over}", hard=True))

    # --- soft quality checks ---
    checks.append(Check("song_count", len(songs) >= target, f"{len(songs)} of {target} songs requested"))

    if profile.genres and songs:
        in_genre = sum(s.genre in profile.genres for s in songs)
        # Don't demand more genre matches than the catalog can supply.
        genre_supply = sum(s.genre in profile.genres for s in pool)
        needed = min(round(MIN_GENRE_SHARE * len(songs)), genre_supply)
        checks.append(Check("genre_fit", in_genre >= needed,
                            f"{in_genre}/{len(songs)} songs in {profile.genres} (need {needed})"))

    if playlist:
        weak_cutoff = WEAK_SCORE_RATIO * max_possible_score(profile)
        weak = [item for item in playlist if item.score < weak_cutoff]
        allowed = int(MAX_WEAK_SHARE * len(playlist))
        checks.append(Check("score_quality", len(weak) <= allowed,
                            f"{len(weak)} weak pick(s) scoring under {weak_cutoff:.2f} (allowed {allowed})"))

        gap = mean(abs(s.energy - profile.target_energy) for s in songs)
        checks.append(Check("energy_fit", gap <= MAX_ENERGY_GAP,
                            f"average energy is {gap:.2f} away from target {profile.target_energy:.2f} "
                            f"(allowed {MAX_ENERGY_GAP})"))

    return Evaluation(checks=checks)


def revise(strategy: Strategy, evaluation: Evaluation, request: ParsedRequest) -> Tuple[Optional[Strategy], str]:
    """RULE-BASED FALLBACK, used only when the AI critic (agent/critic.py) can't be reached.

    Pick a fix for the failed checks. Returns (new_strategy, reason), or (None, reason) if nothing would help.
    Ladder of fixes, cheapest first:
      1. Genre-targeted retrieval: the semantic search drifted away from the requested genres.
      2. Widen the candidate pool: good matches exist but fell outside the top-N retrieved.
    """
    failed = {c.name for c in evaluation.failed}
    quality_issue = failed & {"genre_fit", "score_quality", "energy_fit", "song_count"}
    if not quality_issue:
        return None, "only guardrail checks failed; the final step removes offending songs"

    genres = strategy.profile.genres
    if genres and not strategy.genre_targeted and failed & {"genre_fit", "score_quality"}:
        return (replace(strategy, genre_targeted=True),
                f"{', '.join(sorted(quality_issue))} failed: add a search inside each requested genre "
                f"({', '.join(genres)})")

    if strategy.n_candidates < MAX_CANDIDATES:
        n = min(strategy.n_candidates * 2, MAX_CANDIDATES)
        return (replace(strategy, n_candidates=n),
                f"{', '.join(sorted(quality_issue))} failed: widen the candidate pool "
                f"from {strategy.n_candidates} to {n} songs")

    return None, "candidate pool is already at its maximum; no further fix available"
