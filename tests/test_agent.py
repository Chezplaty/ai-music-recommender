"""Agent reliability tests (plan section 6B), with the AI critic switched off.

No Gemini calls: preferences are passed in directly (or via a fake extractor), and
retrieval uses the fake embedder from conftest.py. So these test the agent's own
behavior: planning, the guardrail checks, the rule-based fallback revision loop,
and the final guardrails.
"""

from dataclasses import replace

import pytest

from preferences import ParsedRequest
from recommender import UserProfile
from guardrails.validation import duplicate_songs, primary_artist, validate_playlist
from rag.retrieval import is_excluded
from agent.evaluator import evaluate
from agent.planner import make_plan
from agent.state import PlaylistItem
from agent.workflow import MAX_REVISIONS, run_agent


def make_request(**overrides) -> ParsedRequest:
    data = dict(
        profile=UserProfile(genres=["metal"], moods=["intense", "angry"], target_energy=0.9,
                            target_valence=0.3, target_acousticness=0.1),
        num_songs=5,
        search_description="heavy distorted guitars, dark and intense metal, loud",
    )
    data.update(overrides)
    return ParsedRequest(**data)


def run(request, catalog, retriever):
    # Rules only: these tests pin down the guardrails and the fallback reviser.
    # The AI critic has its own tests in test_critic.py.
    return run_agent("test", catalog, retriever, request=request, ai_review=False)


def songs_of(result):
    return [item.song for item in result.playlist]


# --- basic behavior --------------------------------------------------------------

def test_returns_requested_number_of_songs(catalog, retriever):
    result = run(make_request(num_songs=8), catalog, retriever)
    assert len(result.playlist) == 8


def test_every_song_is_a_real_catalog_entry(catalog, retriever):
    result = run(make_request(num_songs=10), catalog, retriever)
    assert validate_playlist(songs_of(result), catalog)


def test_no_duplicates(catalog, retriever):
    result = run(make_request(num_songs=15), catalog, retriever)
    assert duplicate_songs(songs_of(result)) == []


def test_respects_genre_constraint(catalog, retriever):
    result = run(make_request(num_songs=5), catalog, retriever)
    assert sum(s.genre == "metal" for s in songs_of(result)) >= 3
    assert result.evaluation.passed


def test_excluded_artists_never_appear(catalog, retriever):
    metal_artists = [s.artist for s in catalog if s.genre == "metal"][:4]
    result = run(make_request(exclude_artists=metal_artists), catalog, retriever)
    assert not any(is_excluded(s, metal_artists) for s in songs_of(result))


def test_at_most_two_songs_per_artist(catalog, retriever):
    result = run(make_request(num_songs=20, profile=UserProfile(genres=[], moods=[], target_energy=0.5)),
                 catalog, retriever)
    counts = {}
    for s in songs_of(result):
        counts[primary_artist(s)] = counts.get(primary_artist(s), 0) + 1
    assert max(counts.values()) <= 2


def test_discover_new_artists_leaves_out_reference_artists(catalog, retriever):
    reference = next(s.artist for s in catalog if s.genre == "metal")
    req = make_request(reference_artists=[reference], discover_new_artists=True)
    result = run(req, catalog, retriever)
    assert not any(is_excluded(s, [reference]) for s in songs_of(result))
    artists = [primary_artist(s) for s in songs_of(result)]
    assert len(artists) == len(set(artists))  # discovery mode: one song per artist


def test_unknown_reference_artist_is_explained(catalog, retriever):
    result = run(make_request(reference_artists=["Definitely Not In Catalog Band"]), catalog, retriever)
    assert any("Definitely Not In Catalog Band" in note for note in result.notes)


def test_fewer_songs_than_requested_when_catalog_runs_out(catalog, retriever):
    """Exclude every artist except a few: the agent returns what's left instead of padding or failing."""
    keep_artists = {s.artist for s in catalog[:3]}
    everyone_else = sorted({s.artist for s in catalog} - keep_artists)
    available = [s for s in catalog if not is_excluded(s, everyone_else)]
    assert 0 < len(available) < 10

    result = run(make_request(num_songs=10, exclude_artists=everyone_else), catalog, retriever)
    assert len(result.playlist) == len(available)
    assert validate_playlist(songs_of(result), catalog)


# --- the revision loop -------------------------------------------------------------

def test_revises_when_search_drifts_from_requested_genre(catalog, retriever):
    """Search text says folk, profile says metal: the first pool has no metal, so the critic
    fails genre_fit and the agent adds a genre-filtered search."""
    req = make_request(search_description="gentle acoustic folk guitar, soft vocals, calm and warm")
    result = run(req, catalog, retriever)
    assert len(result.attempts) >= 2
    assert not result.attempts[0].evaluation.passed
    assert result.attempts[1].strategy.genre_targeted
    assert result.evaluation.passed
    assert any(t["step"] == "revise" for t in result.trace)


def test_stops_after_max_revisions_and_reports_why(catalog, retriever):
    """Metal at energy 0.0 is impossible here (all metal is loud): energy_fit can never pass."""
    req = make_request(profile=UserProfile(genres=["metal"], moods=["sad"], target_energy=0.0))
    result = run(req, catalog, retriever)
    assert len(result.attempts) <= MAX_REVISIONS + 1
    assert result.playlist  # still returns the best attempt instead of nothing
    assert any("energy_fit" in note for note in result.notes)


def test_returns_best_attempt_not_last(catalog, retriever):
    req = make_request(profile=UserProfile(genres=["metal"], moods=["sad"], target_energy=0.0))
    result = run(req, catalog, retriever)
    best_failed = min(len(a.evaluation.failed) for a in result.attempts)
    assert len(result.evaluation.failed) == best_failed


def test_trace_follows_the_workflow(catalog, retriever):
    steps = [t["step"] for t in run(make_request(), catalog, retriever).trace]
    order = ["analyze", "plan", "retrieve", "generate", "evaluate", "finalize"]
    assert [s for s in dict.fromkeys(steps)] == order  # first occurrences in this order


def test_extractor_is_called_with_the_query(catalog, retriever):
    calls = []

    def fake_extractor(query, cat):
        calls.append(query)
        return make_request()

    result = run_agent("loud metal please", catalog, retriever, extractor=fake_extractor, ai_review=False)
    assert calls == ["loud metal please"] and result.playlist


# --- critic and guardrails, directly ---------------------------------------------

def item(song, score=0.9):
    return PlaylistItem(song=song, score=score, reasons=[], similarity=None, confidence="high")


def test_critic_flags_duplicates_exclusions_and_fake_songs(catalog):
    req = make_request(num_songs=4, exclude_artists=[catalog[1].artist])
    strategy, _ = make_plan(req, catalog)
    fake = replace(catalog[2], id=99999, title="Invented Song")
    evaluation = evaluate([item(catalog[0]), item(catalog[0]), item(catalog[1]), item(fake)], req, strategy, catalog)
    failed = {c.name for c in evaluation.failed}
    assert {"in_catalog", "no_duplicates", "excluded_artists"} <= failed


def test_validate_playlist_catches_wrong_title_on_real_id(catalog):
    tampered = replace(catalog[0], title="Not The Real Title")
    assert validate_playlist(catalog[:5], catalog)
    assert not validate_playlist([tampered], catalog)


def test_duplicate_detection_ignores_version_suffixes(catalog):
    live = replace(catalog[0], id=88888, title=catalog[0].title + " - Live At Wembley")
    assert duplicate_songs([catalog[0], live]) == [live]


def test_any_genre_request_is_not_judged_as_weak(catalog, retriever):
    """With no genres/moods the max score is 0.55, so the weak-pick cutoff must scale down."""
    req = make_request(profile=UserProfile(genres=[], moods=[], target_energy=0.9),
                       search_description="loud energetic music")
    result = run(req, catalog, retriever)
    quality = next(c for c in result.evaluation.checks if c.name == "score_quality")
    assert quality.passed
