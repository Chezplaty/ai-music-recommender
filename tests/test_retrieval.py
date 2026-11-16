"""Tests for the RAG layer (src/rag/).

These use a tiny bag-of-words FakeEmbedder instead of the real sentence-transformers
model, so they run offline in about a second and test OUR logic (documents, indexing,
filtering, guardrails, retrieve-then-rank), not the embedding model's quality.
"""

import hashlib
import math
import re

import chromadb
import pytest
from chromadb.config import Settings

from preferences import ParsedRequest
from recommender import UserProfile
from rag.ingestion import ensure_index, load_genre_knowledge, song_to_document
from rag.pipeline import build_search_text, recommend
from rag.retrieval import MusicRetriever, build_retriever, is_excluded

DIM = 512


class FakeEmbedder:
    """Hashes words into a fixed-size, normalized count vector. Shared words -> higher cosine."""

    def __init__(self, name="fake-bow"):
        self.name = name
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            v = [0.0] * DIM
            for word in re.findall(r"[a-z&]+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


def make_client(path):
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


@pytest.fixture
def retriever(tmp_path, catalog):
    return build_retriever(catalog, embedder=FakeEmbedder(), chroma_client=make_client(tmp_path))


def make_request(**overrides) -> ParsedRequest:
    data = dict(
        profile=UserProfile(genres=["metal"], moods=["intense"], target_energy=0.9),
        num_songs=5,
        search_description="heavy distorted guitars, dark and intense metal",
    )
    data.update(overrides)
    return ParsedRequest(**data)


# --- documents ----------------------------------------------------------------

def test_song_document_has_metadata_descriptors_and_genre_knowledge(catalog):
    song = catalog[0]
    doc = song_to_document(song, load_genre_knowledge())
    assert f"Song: {song.title}" in doc
    assert f"Artist: {song.artist}" in doc
    assert f"Genre: {song.genre}" in doc
    assert f"Mood: {song.mood}" in doc
    assert "Sound:" in doc and "BPM" in doc
    assert "Genre characteristics:" in doc  # enriched from data/music_knowledge.json


def test_every_catalog_genre_has_knowledge(catalog):
    knowledge = load_genre_knowledge()
    assert {s.genre for s in catalog} <= set(knowledge)


# --- indexing -----------------------------------------------------------------

def test_index_contains_every_catalog_song(retriever, catalog):
    assert retriever.collection.count() == len(catalog)


def test_index_is_reused_when_nothing_changed(tmp_path, catalog):
    client, embedder = make_client(tmp_path), FakeEmbedder()
    _, rebuilt_first = ensure_index(client, catalog, embedder)
    _, rebuilt_second = ensure_index(client, catalog, embedder)
    assert rebuilt_first and not rebuilt_second


def test_index_rebuilds_when_catalog_changes(tmp_path, catalog):
    client, embedder = make_client(tmp_path), FakeEmbedder()
    ensure_index(client, catalog, embedder)
    collection, rebuilt = ensure_index(client, catalog[:50], embedder)
    assert rebuilt and collection.count() == 50


# --- search -------------------------------------------------------------------

def test_search_returns_catalog_songs_sorted_by_similarity(retriever, catalog):
    results = retriever.search("calm acoustic folk guitar", n_results=10)
    assert len(results) == 10
    catalog_ids = {s.id for s in catalog}
    assert all(r.song.id in catalog_ids for r in results)
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)


def test_search_finds_the_requested_genre(retriever):
    results = retriever.search("metal: heavily distorted guitars, double-kick drums, screamed vocals", n_results=5)
    assert sum(r.song.genre == "metal" for r in results) >= 3


def test_search_respects_excluded_artists(retriever, catalog):
    artist = catalog[0].artist
    results = retriever.search(song_to_document(catalog[0], load_genre_knowledge()),
                               n_results=50, exclude_artists=[artist.upper()])
    assert len(results) == 50  # exclusions are filtered in Chroma, not after, so the pool stays full
    assert all(artist.lower() not in r.song.artist.lower() for r in results)


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("   ") == []


def test_ids_missing_from_catalog_are_never_returned(tmp_path, catalog):
    """Guardrail: an index built from a bigger catalog must not leak songs we no longer have."""
    embedder = FakeEmbedder()
    collection, _ = ensure_index(make_client(tmp_path), catalog, embedder)
    subset = catalog[:20]
    results = MusicRetriever(collection, subset, embedder).search("pop", n_results=100)
    assert {r.song.id for r in results} <= {s.id for s in subset}


def test_is_excluded_is_case_insensitive_substring():
    from recommender import Song
    song = Song(1, "t", "The Weeknd, Daft Punk", "pop", "happy", 0.5, 120, 0.5, 0.5, 0.5)
    assert is_excluded(song, ["weeknd"])
    assert not is_excluded(song, ["drake", ""])


# --- retrieve-then-rank pipeline ---------------------------------------------

def test_search_text_prefers_sound_description_and_adds_reference_artists():
    req = make_request(search_description="dreamy r&b", reference_artists=["Frank Ocean"])
    assert build_search_text(req, query="I'm studying") == "dreamy r&b\nSimilar to: Frank Ocean"
    assert build_search_text(make_request(search_description=""), query="I'm studying") == "I'm studying"


def test_rag_playlist_only_contains_retrieved_candidates(retriever, catalog):
    result = recommend(make_request(), catalog, retriever=retriever)
    assert result.mode == "rag"
    assert len(result.results) == 5
    candidate_ids = {r.song.id for r in result.retrieved}
    assert all(song.id in candidate_ids for song, _, _ in result.results)
    scores = [score for _, score, _ in result.results]
    assert scores == sorted(scores, reverse=True)  # ranked by the original scorer


def test_rag_playlist_respects_excluded_artists(retriever, catalog):
    metal_artists = [s.artist for s in catalog if s.genre == "metal"]
    result = recommend(make_request(exclude_artists=metal_artists[:3]), catalog, retriever=retriever)
    assert not any(is_excluded(song, metal_artists[:3]) for song, _, _ in result.results)


def test_without_retriever_runs_original_system(catalog):
    result = recommend(make_request(num_songs=7), catalog, retriever=None)
    assert result.mode == "baseline" and result.retrieved == []
    assert len(result.results) == 7
