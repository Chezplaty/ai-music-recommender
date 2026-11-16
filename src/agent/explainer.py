"""
Grounded explanations: why each song is in the playlist. No LLM involved.

  build_evidence()       lists facts about each song from real data only: catalog
                         features, the scorer's breakdown, the retrieval match.
                         Each fact is a reason (supports=True), a trade-off
                         (supports=False), or neutral info (supports=None).
  explain_song()         turns the strongest reasons and trade-offs into 1-3 sentences.
  summarize_playlist()   one short paragraph about the playlist as a whole.

Because every sentence is assembled from a Fact, every claim (and every number)
in an explanation is traceable to the data, and `cited` lists which facts were used.
"""

from collections import Counter
from dataclasses import replace
from statistics import mean
from typing import List, Tuple

from preferences import ParsedRequest
from rag.ingestion import describe_acousticness, describe_danceability, describe_tempo, describe_valence
from agent.state import Fact, PlaylistItem, Strategy

MAX_REASONS = 3
MAX_TRADEOFFS = 2
CLOSE = 0.2  # within 0.2 of the target counts as a reason; further is a trade-off

# Order reasons are mentioned in (most meaningful to a listener first).
REASON_ORDER = ["reference_artist", "genre", "mood", "energy", "valence", "acousticness"]


def _closeness(diff: float) -> str:
    return "very close to" if diff <= 0.1 else "close to" if diff <= CLOSE else "well off"


def _or_list(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def build_evidence(item: PlaylistItem, strategy: Strategy, request: ParsedRequest) -> List[Fact]:
    """Facts about one song, measured against the preferences the playlist was scored with."""
    s, p = item.song, strategy.profile
    facts = []

    refs = [a for a in request.reference_artists if a.lower() in s.artist.lower()]
    if refs:
        facts.append(Fact("reference_artist", f"it's by {', '.join(refs)}, who you mentioned", True))

    if p.genres:
        hit = s.genre in p.genres
        facts.append(Fact("genre", f"it's {s.genre}, one of your genres" if hit
                          else f"it's {s.genre} rather than {_or_list(p.genres)}", hit))

    if p.moods:
        hit = s.mood in p.moods
        facts.append(Fact("mood", f"its {s.mood} mood fits what you asked for" if hit
                          else f"its mood is {s.mood} rather than {_or_list(p.moods)}", hit))

    for name, value, target, describe in (
        ("energy", s.energy, p.target_energy, None),
        ("valence", s.valence, p.target_valence, describe_valence),
        ("acousticness", s.acousticness, p.target_acousticness, describe_acousticness),
    ):
        diff = abs(value - target)
        label = {"energy": "energy", "valence": "positivity (valence)", "acousticness": "acousticness"}[name]
        feel = f" ({describe(value)})" if describe else ""
        facts.append(Fact(name, f"its {label} of {value:.2f} is {_closeness(diff)} your target of {target:.2f}{feel}",
                          diff <= CLOSE))

    facts.append(Fact("tempo", describe_tempo(s.tempo_bpm), None))
    facts.append(Fact("danceability", f"{describe_danceability(s.danceability)} ({s.danceability:.2f})", None))

    if item.similarity is not None:
        strength = "closely" if item.similarity >= 0.55 else "reasonably" if item.similarity >= 0.45 else "loosely"
        facts.append(Fact("search_match", f"it {strength} matches the sound we searched for "
                                          f"(similarity {item.similarity:.2f})", item.similarity >= 0.45))
    if s.album:
        facts.append(Fact("album", f"from the album '{s.album}'", None))
    facts.append(Fact("score", f"overall score {item.score:.2f} ({item.confidence} confidence)", None))
    return facts


def _sentence(phrases: List[str]) -> str:
    if len(phrases) > 1:
        phrases = phrases[:-1] + ["and " + phrases[-1]]
    text = ", ".join(phrases) if len(phrases) > 2 else " ".join(phrases)
    return text[0].upper() + text[1:] + "."


def explain_song(evidence: List[Fact]) -> Tuple[str, List[str]]:
    """1-3 sentences: top reasons, the search match, and the biggest trade-offs. Returns (text, cited fact ids)."""
    by_id = {f.id: f for f in evidence}
    reasons = [by_id[i] for i in REASON_ORDER if i in by_id and by_id[i].supports][:MAX_REASONS]
    tradeoffs = [by_id[i] for i in REASON_ORDER if i in by_id and by_id[i].supports is False][:MAX_TRADEOFFS]
    match = by_id.get("search_match")

    sentences = []
    if reasons:
        sentences.append(_sentence([f.text for f in reasons]))
    if match is not None:
        sentences.append(match.text[0].upper() + match.text[1:] + ".")
    if tradeoffs:
        sentences.append("Trade-off: " + "; ".join(f.text for f in tradeoffs) + ".")
    if not sentences:
        sentences.append(by_id["score"].text.capitalize() + ".")

    cited = [f.id for f in reasons] + (["search_match"] if match else []) + [f.id for f in tradeoffs]
    return " ".join(sentences), cited


def summarize_playlist(playlist: List[PlaylistItem], strategy: Strategy) -> str:
    """A short, factual overview: genre mix, energy vs target, confidence, artist variety."""
    if not playlist:
        return ""
    p = strategy.profile
    genres = Counter(i.song.genre for i in playlist).most_common()
    genre_mix = ", ".join(f"{n} {g}" for g, n in genres)
    energy = mean(i.song.energy for i in playlist)
    high = sum(i.confidence == "high" for i in playlist)
    artists = len({i.song.artist for i in playlist})
    return (f"{len(playlist)} songs ({genre_mix}) from {artists} artists. Average energy {energy:.2f} "
            f"against your target of {p.target_energy:.2f}; {high} of {len(playlist)} picks are high-confidence matches.")


def explain_playlist(request: ParsedRequest, strategy: Strategy,
                     playlist: List[PlaylistItem]) -> Tuple[List[PlaylistItem], str]:
    """Returns (playlist with evidence + explanations, playlist summary)."""
    out = []
    for item in playlist:
        evidence = build_evidence(item, strategy, request)
        text, cited = explain_song(evidence)
        out.append(replace(item, evidence=evidence, explanation=text, cited=cited))
    return out, summarize_playlist(out, strategy)
