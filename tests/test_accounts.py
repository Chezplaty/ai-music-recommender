"""Tests for local accounts and taste profiles (src/accounts.py)."""

import json

import pytest

from accounts import AccountError, TasteProfile, UserStore, parse_artist_list


@pytest.fixture
def store(tmp_path):
    return UserStore(tmp_path / "users.json")


def test_create_and_log_in(store):
    assert store.create_user("Mai_Linh", "secret123") == "mai_linh"  # usernames are case-insensitive
    assert store.authenticate("MAI_LINH", "secret123") == "mai_linh"


def test_password_is_never_stored_in_plain_text(store):
    store.create_user("alice", "secret123")
    raw = store.path.read_text()
    assert "secret123" not in raw
    record = json.loads(raw)["alice"]
    assert set(record) >= {"salt", "password_hash"}


def test_same_password_gets_different_hashes(store):
    store.create_user("alice", "secret123")
    store.create_user("bob", "secret123")
    data = json.loads(store.path.read_text())
    assert data["alice"]["password_hash"] != data["bob"]["password_hash"]  # random salt per user


def test_wrong_password_and_unknown_user_give_the_same_error(store):
    store.create_user("alice", "secret123")
    with pytest.raises(AccountError) as wrong:
        store.authenticate("alice", "nope")
    with pytest.raises(AccountError) as unknown:
        store.authenticate("nobody", "secret123")
    assert str(wrong.value) == str(unknown.value)


@pytest.mark.parametrize("username,password", [("ab", "secret123"), ("bad name!", "secret123"), ("alice", "123")])
def test_invalid_signups_are_rejected(store, username, password):
    with pytest.raises(AccountError):
        store.create_user(username, password)


def test_duplicate_username_is_rejected(store):
    store.create_user("alice", "secret123")
    with pytest.raises(AccountError, match="taken"):
        store.create_user("Alice", "other456")


def test_profile_round_trip(store):
    store.create_user("alice", "secret123")
    assert store.get_profile("alice") is None  # new users have no profile yet
    profile = TasteProfile(display_name="Alice", genres=["r&b"], moods=["chill"], energy=0.4,
                           favorite_artists=["Frank Ocean"], avoid_artists=["Nickelback"])
    store.save_profile("alice", profile)
    assert UserStore(store.path).get_profile("ALICE") == profile


def test_profile_to_request_needs_no_llm():
    profile = TasteProfile(display_name="A", genres=["jazz"], moods=["relaxed"], energy=0.3, valence=0.7,
                           acousticness=0.8, favorite_artists=["Norah Jones"], avoid_artists=["X"])
    req = profile.to_request(num_songs=7)
    assert req.num_songs == 7
    assert req.profile.genres == ["jazz"] and req.profile.target_energy == 0.3
    assert req.reference_artists == ["Norah Jones"] and req.exclude_artists == ["X"]
    assert "jazz" in req.search_description and "acoustic" in req.search_description


def test_profile_prompt_text_lists_tastes():
    text = TasteProfile(display_name="A", genres=["metal"], avoid_artists=["Drake"]).to_prompt_text()
    assert "favorite genres: metal" in text and "never play: Drake" in text


def test_parse_artist_list():
    assert parse_artist_list("Frank Ocean, SZA\nfrank ocean,, The Weeknd ") == ["Frank Ocean", "SZA", "The Weeknd"]
    assert parse_artist_list("") == []
