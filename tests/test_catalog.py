"""Data-quality checks on data/catalog.csv. The agent's guardrails will rely on
song IDs being unique and features being in range, so we verify that here."""

from collections import Counter

FEATURES = ["energy", "valence", "danceability", "acousticness"]


def test_catalog_has_300_songs(catalog):
    assert len(catalog) == 300


def test_ids_are_unique(catalog):
    assert len({s.id for s in catalog}) == len(catalog)
    assert len({s.spotify_track_id for s in catalog}) == len(catalog)


def test_no_duplicate_songs(catalog):
    keys = [(s.artist.lower(), s.title.lower()) for s in catalog]
    assert len(set(keys)) == len(keys)


def test_features_are_in_range(catalog):
    for song in catalog:
        for name in FEATURES:
            assert 0.0 <= getattr(song, name) <= 1.0, (song.title, name)
        assert song.tempo_bpm > 0


def test_genres_are_balanced(catalog):
    counts = Counter(s.genre for s in catalog)
    assert len(counts) == 25
    assert set(counts.values()) == {12}


def test_every_song_has_required_text_and_source(catalog):
    for song in catalog:
        assert song.title and song.artist and song.genre and song.mood
        assert song.source in {"tidytuesday", "spotify-tracks-hf"}
