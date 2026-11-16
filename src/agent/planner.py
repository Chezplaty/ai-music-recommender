"""Turn a ParsedRequest into a Strategy: what to search for, how wide, and which artist rules apply."""

from typing import List, Tuple

from preferences import ParsedRequest
from recommender import Song
from rag.pipeline import build_search_text
from agent.state import Strategy

MIN_CANDIDATES = 30
CANDIDATES_PER_SONG = 3
MAX_PER_ARTIST = 2             # variety: no artist more than twice
MAX_PER_ARTIST_DISCOVERY = 1   # "discover new artists": one song each


def artist_in_catalog(name: str, catalog: List[Song]) -> bool:
    name = name.lower()
    return any(name in s.artist.lower() for s in catalog)


def make_plan(request: ParsedRequest, catalog: List[Song], query: str = "") -> Tuple[Strategy, List[str]]:
    """Returns (strategy, notes). Notes are limitations the user should hear about."""
    notes = []
    exclude = list(request.exclude_artists)
    if request.discover_new_artists and request.reference_artists:
        exclude += [a for a in request.reference_artists if a not in exclude]

    missing = [a for a in request.reference_artists if not artist_in_catalog(a, catalog)]
    if missing:
        notes.append(f"Not in the catalog: {', '.join(missing)}. Searching by their described sound instead.")
    if request.unmatched:
        notes.append(f"Couldn't match from the catalog: {', '.join(request.unmatched)}.")

    max_per_artist = MAX_PER_ARTIST_DISCOVERY if request.discover_new_artists else MAX_PER_ARTIST
    n = max(MIN_CANDIDATES, CANDIDATES_PER_SONG * request.num_songs)
    search_text = build_search_text(request, query)

    steps = [
        f"Retrieve the {n} songs whose descriptions best match: \"{search_text.splitlines()[0][:90]}\"",
        "Rank those candidates with the weighted scorer (genre, mood, energy, valence, acousticness)",
        f"Pick the top {request.num_songs}, at most {max_per_artist} per artist"
        + (f", leaving out {', '.join(exclude)}" if exclude else ""),
        "Check the playlist against the request; revise the search if it falls short",
    ]
    strategy = Strategy(search_text=search_text, n_candidates=n, exclude_artists=exclude,
                        max_per_artist=max_per_artist, steps=steps)
    return strategy, notes
