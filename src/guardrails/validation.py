"""
Hard guardrails for playlists. Pure functions, no AI involved.

These run in two places: the agent's critic (to decide whether to revise) and
the final step (to strip anything that slipped through, so the user never sees an
invented, duplicated, or excluded song).
"""

import re
from typing import Iterable, List

from recommender import Song


def primary_artist(song: Song) -> str:
    """'Drake, Future' -> 'drake'. Used for per-artist limits."""
    return song.artist.split(",")[0].strip().lower()


def normalize_title(title: str) -> str:
    """'Valerie - Live At BBC Radio 1' / 'Valerie (feat. X)' -> 'valerie'."""
    title = re.sub(r"\s*[\(\[].*?[\)\]]", "", title.lower())
    title = re.sub(r"\s+-\s+.*$", "", title)
    return title.strip()


def song_key(song: Song) -> tuple:
    """Two catalog rows with the same key are the same song (e.g. a live version)."""
    return primary_artist(song), normalize_title(song.title)


def invalid_songs(playlist: Iterable[Song], catalog: List[Song]) -> List[Song]:
    """Songs that don't match a catalog row exactly (by id AND title/artist).

    Checking the content too, not just the id, catches an id that was
    attached to the wrong song somewhere along the way.
    """
    by_id = {s.id: s for s in catalog}
    return [s for s in playlist
            if s.id not in by_id or (by_id[s.id].title, by_id[s.id].artist) != (s.title, s.artist)]


def validate_playlist(playlist: Iterable[Song], catalog: List[Song]) -> bool:
    """True if every song is a real catalog entry. The anti-hallucination check."""
    return not invalid_songs(playlist, catalog)


def duplicate_songs(playlist: Iterable[Song]) -> List[Song]:
    """Every song after the first occurrence of the same id or same artist+title."""
    seen_ids, seen_keys, dupes = set(), set(), []
    for s in playlist:
        if s.id in seen_ids or song_key(s) in seen_keys:
            dupes.append(s)
        seen_ids.add(s.id)
        seen_keys.add(song_key(s))
    return dupes
