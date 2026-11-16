"""Tests for NL preference extraction. No real API calls: a fake client returns
canned LLM output, so these run offline and test OUR guardrails, not Gemini."""

import json
from types import SimpleNamespace

import pytest

from preferences import (
    ExtractedPreferences, ExtractionError, build_vocabulary, extract_preferences,
    validate_preferences, MAX_NUM_SONGS, DEFAULT_NUM_SONGS,
)


def make_extracted(**overrides) -> ExtractedPreferences:
    data = dict(
        genres=["pop"], moods=["happy"], target_energy=0.7, target_valence=0.6,
        target_acousticness=0.3, num_songs=10, reference_artists=[], exclude_artists=[],
        unmatched_requests=[], search_description="upbeat electronic pop", discover_new_artists=False,
    )
    data.update(overrides)
    return ExtractedPreferences(**data)


class FakeClient:
    """Mimics client.models.generate_content() and records what it was sent."""

    def __init__(self, parsed=None, text=""):
        self.calls = []
        self._response = SimpleNamespace(parsed=parsed, text=text)
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


@pytest.fixture
def vocab(catalog):
    return build_vocabulary(catalog)


# --- validate_preferences (the guardrail) ------------------------------------

def test_out_of_vocabulary_genres_and_moods_move_to_unmatched(vocab):
    parsed = validate_preferences(make_extracted(genres=["pop", "vaporwave"], moods=["happy", "spooky"]), vocab)
    assert parsed.profile.genres == ["pop"]
    assert parsed.profile.moods == ["happy"]
    assert "vaporwave" in parsed.unmatched and "spooky" in parsed.unmatched


def test_genre_case_and_whitespace_are_normalized(vocab):
    parsed = validate_preferences(make_extracted(genres=[" Hip Hop ", "hip hop"]), vocab)
    assert parsed.profile.genres == ["hip hop"]  # normalized and deduped


def test_energy_valence_and_acousticness_are_clamped(vocab):
    parsed = validate_preferences(
        make_extracted(target_energy=1.7, target_valence=-0.3, target_acousticness=2.0), vocab
    )
    assert parsed.profile.target_energy == 1.0
    assert parsed.profile.target_valence == 0.0
    assert parsed.profile.target_acousticness == 1.0


@pytest.mark.parametrize("asked, expected", [(0, DEFAULT_NUM_SONGS), (-5, DEFAULT_NUM_SONGS), (5, 5), (500, MAX_NUM_SONGS)])
def test_num_songs_is_bounded(vocab, asked, expected):
    assert validate_preferences(make_extracted(num_songs=asked), vocab).num_songs == expected


def test_llm_reported_unmatched_requests_are_kept(vocab):
    parsed = validate_preferences(make_extracted(unmatched_requests=["songs from the 1920s"]), vocab)
    assert parsed.unmatched == ["songs from the 1920s"]


# --- extract_preferences (with a fake LLM) -----------------------------------

def test_extract_uses_parsed_response(catalog):
    client = FakeClient(parsed=make_extracted(genres=["jazz"], exclude_artists=["Drake"]))
    parsed = extract_preferences("smooth jazz, no Drake", catalog, client=client, model="test-model")

    assert parsed.profile.genres == ["jazz"]
    assert parsed.exclude_artists == ["Drake"]
    assert parsed.raw_llm_output["genres"] == ["jazz"]


def test_prompt_contains_catalog_vocabulary_and_schema(catalog, vocab):
    client = FakeClient(parsed=make_extracted())
    extract_preferences("anything", catalog, client=client, model="test-model")

    call = client.calls[0]
    assert call["contents"] == "anything"
    assert call["config"].response_schema is ExtractedPreferences
    for genre in vocab["genres"]:
        assert genre in call["config"].system_instruction


def test_falls_back_to_raw_json_text_when_parsed_is_missing(catalog):
    text = json.dumps(make_extracted(genres=["metal"]).model_dump())
    parsed = extract_preferences("heavy stuff", catalog, client=FakeClient(parsed=None, text=text), model="m")
    assert parsed.profile.genres == ["metal"]


def test_unparseable_response_raises(catalog):
    with pytest.raises(ExtractionError):
        extract_preferences("hi", catalog, client=FakeClient(parsed=None, text="not json"), model="m")


def test_empty_query_raises_without_calling_llm(catalog):
    client = FakeClient(parsed=make_extracted())
    with pytest.raises(ExtractionError):
        extract_preferences("   ", catalog, client=client, model="m")
    assert client.calls == []


def test_missing_api_key_gives_clear_error(catalog, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ExtractionError, match="GEMINI_API_KEY"):
        extract_preferences("anything", catalog)


def test_listener_profile_is_sent_with_the_request(catalog):
    client = FakeClient(parsed=make_extracted())
    extract_preferences("something for the gym", catalog, client=client, model="m",
                        listener_profile="favorite genres: metal")
    contents = client.calls[0]["contents"]
    assert "favorite genres: metal" in contents and "REQUEST: something for the gym" in contents


def test_no_profile_sends_the_request_as_is(catalog):
    client = FakeClient(parsed=make_extracted())
    extract_preferences("something for the gym", catalog, client=client, model="m")
    assert client.calls[0]["contents"] == "something for the gym"
