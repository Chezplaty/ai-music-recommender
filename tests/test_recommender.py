from dataclasses import asdict

import pytest

from recommender import Song, UserProfile, Recommender, score_song, recommend_songs

POP_FAN = UserProfile(genres=["pop"], moods=["happy"], target_energy=0.8, target_acousticness=0.0)


def make_song(id, genre, mood, energy, valence=0.5, acousticness=0.5) -> Song:
    return Song(
        id=id, title=f"Song {id}", artist="Test Artist", genre=genre, mood=mood,
        energy=energy, tempo_bpm=100, valence=valence, danceability=0.5,
        acousticness=acousticness,
    )


def make_small_recommender() -> Recommender:
    # Lofi song listed FIRST so a recommender that ignores scores would fail.
    return Recommender([
        make_song(2, "lofi", "chill", energy=0.4, valence=0.6, acousticness=0.9),
        make_song(1, "pop", "happy", energy=0.8, valence=0.9, acousticness=0.2),
    ])


# --- Recommender class -------------------------------------------------------

def test_recommend_returns_songs_sorted_by_score():
    results = make_small_recommender().recommend(POP_FAN, k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_recommend_respects_k():
    rec = make_small_recommender()
    assert len(rec.recommend(POP_FAN, k=1)) == 1
    assert len(rec.recommend(POP_FAN, k=10)) == 2  # never more than the catalog


def test_recommend_with_scores_is_sorted_descending():
    results = make_small_recommender().recommend_with_scores(POP_FAN, k=2)
    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True)


def test_class_and_function_interfaces_agree():
    rec = make_small_recommender()
    class_ids = [s.id for s in rec.recommend(POP_FAN, k=2)]
    func_ids = [s["id"] for s, _, _ in recommend_songs(POP_FAN.to_prefs(), [asdict(s) for s in rec.songs], k=2)]
    assert class_ids == func_ids


def test_explain_recommendation_cites_matching_features():
    rec = make_small_recommender()
    pop_song = next(s for s in rec.songs if s.genre == "pop")

    explanation = rec.explain_recommendation(POP_FAN, pop_song)
    assert "Matches favorite genre 'pop'" in explanation
    assert "Matches favorite mood 'happy'" in explanation


# --- score_song --------------------------------------------------------------

def test_perfect_match_scores_one():
    user = UserProfile(genres=["pop"], moods=["happy"], target_energy=0.8, target_valence=0.9, target_acousticness=0.0)
    song = make_song(1, "pop", "happy", energy=0.8, valence=0.9, acousticness=0.0)
    score, _ = score_song(user.to_prefs(), asdict(song))
    assert score == pytest.approx(1.0)


def test_genre_match_is_worth_030():
    user = UserProfile(genres=["pop"], moods=["happy"], target_energy=0.5)
    match = make_song(1, "pop", "sad", energy=0.5)
    miss = make_song(2, "rock", "sad", energy=0.5)
    diff = score_song(user.to_prefs(), asdict(match))[0] - score_song(user.to_prefs(), asdict(miss))[0]
    assert diff == pytest.approx(0.30)


def test_any_listed_genre_counts_as_match():
    user = UserProfile(genres=["pop", "rock"], moods=["happy"], target_energy=0.5)
    score_pop, _ = score_song(user.to_prefs(), asdict(make_song(1, "pop", "sad", 0.5)))
    score_rock, _ = score_song(user.to_prefs(), asdict(make_song(2, "rock", "sad", 0.5)))
    assert score_pop == pytest.approx(score_rock)


def test_mid_acousticness_target_prefers_mid_acoustic_song():
    user = UserProfile(genres=["pop"], moods=["happy"], target_energy=0.5, target_acousticness=0.3)
    near = make_song(1, "pop", "happy", energy=0.5, acousticness=0.3)
    electronic = make_song(2, "pop", "happy", energy=0.5, acousticness=0.0)
    unplugged = make_song(3, "pop", "happy", energy=0.5, acousticness=1.0)
    ranked = Recommender([unplugged, electronic, near]).recommend(user, k=3)
    assert [s.id for s in ranked] == [1, 2, 3]


@pytest.mark.parametrize("likes_acoustic, expected_target", [(True, 1.0), (False, 0.0)])
def test_legacy_likes_acoustic_dict_still_works(likes_acoustic, expected_target):
    # main.py's demo profiles still use the old bool format
    prefs = {"genre": ["pop"], "mood": ["happy"], "energy": 0.5, "likes_acoustic": likes_acoustic}
    song = make_song(1, "pop", "happy", energy=0.5, acousticness=expected_target)
    _, reasons = score_song(prefs, asdict(song))
    assert "Acousticness similarity 1.00" in reasons[-1]


@pytest.mark.parametrize("energy", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("acoustic", [0.0, 0.5, 1.0])
def test_scores_stay_between_zero_and_one(catalog, energy, acoustic):
    user = UserProfile(genres=["pop"], moods=["happy"], target_energy=energy, target_acousticness=acoustic)
    for song in catalog:
        score, _ = score_song(user.to_prefs(), asdict(song))
        assert 0.0 <= score <= 1.0
