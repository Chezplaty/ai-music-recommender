"""Playlist generator: the ORIGINAL scorer ranks the candidates; this applies the playlist rules."""

from collections import Counter
from typing import Dict, List

from preferences import ParsedRequest
from recommender import Recommender, Song, UserProfile
from guardrails.validation import primary_artist, song_key
from rag.retrieval import is_excluded
from agent.state import PlaylistItem, Strategy

HIGH_CONFIDENCE = 0.85  # fraction of the best score this request allows
MEDIUM_CONFIDENCE = 0.70


def max_possible_score(profile: UserProfile) -> float:
    """Best score any song could get for this profile (weights from score_song()).

    With no genres requested, genre_sim is always 0, so 0.30 is unreachable; same
    for mood (0.15). Thresholds are taken relative to this so a "any genre" request
    isn't judged as if every song were a bad match.
    """
    return 1.0 - (0.0 if profile.genres else 0.30) - (0.0 if profile.moods else 0.15)


def confidence(score: float, profile: UserProfile) -> str:
    ratio = score / max_possible_score(profile)
    return "high" if ratio >= HIGH_CONFIDENCE else "medium" if ratio >= MEDIUM_CONFIDENCE else "low"


def build_playlist(candidates: List[Song], request: ParsedRequest, strategy: Strategy,
                   similarity: Dict[int, float]) -> List[PlaylistItem]:
    ranked = Recommender(candidates).recommend_with_scores(request.profile, k=len(candidates))

    playlist, seen_ids, seen_keys, per_artist = [], set(), set(), Counter()
    for song, score, reasons in ranked:
        if len(playlist) >= request.num_songs:
            break
        if song.id in seen_ids or song_key(song) in seen_keys:
            continue  # no duplicates, including other versions of the same song
        if is_excluded(song, strategy.exclude_artists):
            continue
        if per_artist[primary_artist(song)] >= strategy.max_per_artist:
            continue
        seen_ids.add(song.id)
        seen_keys.add(song_key(song))
        per_artist[primary_artist(song)] += 1
        playlist.append(PlaylistItem(song=song, score=score, reasons=reasons,
                                     similarity=similarity.get(song.id),
                                     confidence=confidence(score, request.profile)))
    return playlist
