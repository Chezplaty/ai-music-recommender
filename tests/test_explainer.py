"""Tests for grounded explanations (src/agent/explainer.py). No LLM involved."""

import pytest

from preferences import ParsedRequest
from recommender import UserProfile
from agent.explainer import build_evidence, explain_playlist, explain_song, summarize_playlist
from agent.planner import make_plan
from agent.state import PlaylistItem
from agent.workflow import run_agent


@pytest.fixture
def request_():
    return ParsedRequest(profile=UserProfile(genres=["r&b"], moods=["relaxed"], target_energy=0.45,
                                             target_valence=0.6, target_acousticness=0.3),
                         num_songs=3, search_description="smooth mellow r&b")


@pytest.fixture
def setup(catalog, request_):
    strategy, _ = make_plan(request_, catalog)
    rnb = [s for s in catalog if s.genre == "r&b"][:2]
    metal = next(s for s in catalog if s.genre == "metal")
    playlist = [PlaylistItem(song=s, score=0.9, reasons=[], similarity=0.6, confidence="high") for s in rnb]
    playlist.append(PlaylistItem(song=metal, score=0.5, reasons=[], similarity=None, confidence="low"))
    return request_, strategy, playlist


def facts(item, strategy, request):
    return {f.id: f for f in build_evidence(item, strategy, request)}


# --- evidence -----------------------------------------------------------------------

def test_evidence_comes_from_real_song_data(setup):
    request, strategy, playlist = setup
    song = playlist[0].song
    f = facts(playlist[0], strategy, request)
    assert f["genre"].supports is True and "r&b" in f["genre"].text
    assert f"{song.energy:.2f}" in f["energy"].text and "0.45" in f["energy"].text
    assert "0.60" in f["search_match"].text
    assert {"mood", "valence", "acousticness", "tempo", "danceability", "score"} <= set(f)


def test_mismatches_are_trade_offs(setup):
    request, strategy, playlist = setup
    f = facts(playlist[2], strategy, request)  # the metal song
    assert f["genre"].supports is False and "rather than r&b" in f["genre"].text
    assert "search_match" not in f  # not retrieved, so no match claim


def test_reference_artist_fact_only_when_true(setup):
    request, strategy, playlist = setup
    req = ParsedRequest(**{**request.__dict__, "reference_artists": [playlist[0].song.artist]})
    assert facts(playlist[0], strategy, req)["reference_artist"].supports is True
    assert "reference_artist" not in facts(playlist[2], strategy, req)


def test_evidence_uses_the_strategy_profile_not_the_original(setup):
    """After the AI critic changes the targets, explanations must match what the songs were scored against."""
    request, strategy, playlist = setup
    from dataclasses import replace
    revised = replace(strategy, profile=replace(strategy.profile, target_energy=0.9))
    assert "0.90" in facts(playlist[0], revised, request)["energy"].text


# --- explanations ---------------------------------------------------------------------

def test_explanation_is_built_only_from_cited_facts(setup):
    request, strategy, playlist = setup
    evidence = build_evidence(playlist[0], strategy, request)
    text, cited = explain_song(evidence)
    by_id = {f.id: f for f in evidence}
    assert cited and all(c in by_id for c in cited)
    for c in cited:  # every cited fact's wording appears in the explanation
        assert by_id[c].text[1:] in text


def test_explanation_reads_as_sentences(setup):
    request, strategy, playlist = setup
    text, _ = explain_song(build_evidence(playlist[0], strategy, request))
    assert text[0].isupper() and text.endswith(".")


def test_explanation_mentions_trade_offs(setup):
    request, strategy, playlist = setup
    text, cited = explain_song(build_evidence(playlist[2], strategy, request))
    assert "Trade-off:" in text and "genre" in cited


def test_explanations_are_deterministic(setup):
    request, strategy, playlist = setup
    assert explain_playlist(request, strategy, playlist) == explain_playlist(request, strategy, playlist)


def test_summary_describes_the_playlist(setup):
    request, strategy, playlist = setup
    items, summary = explain_playlist(request, strategy, playlist)
    assert summary.startswith("3 songs (2 r&b, 1 metal)")
    assert "0.45" in summary and "2 of 3 picks are high-confidence" in summary
    assert summarize_playlist([], strategy) == ""


# --- in the workflow -----------------------------------------------------------------

def test_agent_explains_every_final_song(catalog, retriever, request_):
    result = run_agent("chill r&b for studying", catalog, retriever, request=request_, ai_review=False)
    assert result.playlist and all(i.explanation and i.evidence and i.cited for i in result.playlist)
    assert result.summary
    assert any(t["step"] == "explain" for t in result.trace)
