"""Click-through tests for the Streamlit app (app.py), using Streamlit's AppTest.

Accounts go to a temp file. Gemini is never called: the end-to-end test uses
Quick mix with the AI critic off, which runs the real retriever and agent offline.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from accounts import TasteProfile, UserStore

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def users(tmp_path, monkeypatch):
    path = tmp_path / "users.json"
    monkeypatch.setenv("MUSIC_APP_USERS", str(path))
    return UserStore(path)


def start():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def submit(at, label):
    next(b for b in at.button if b.label == label).click()
    at.run()


def test_starts_on_the_login_page(users):
    at = start()
    assert not at.exception
    assert [t.label for t in at.tabs][:2] == ["Log in", "Create account"]


def test_sign_up_leads_to_profile_setup_then_the_app(users):
    at = start()
    at.text_input(key="signup_user").input("newuser")
    at.text_input(key="signup_pass").input("secret123")
    at.text_input(key="signup_confirm").input("secret123")
    submit(at, "Create account")
    assert not at.exception
    assert any("get to know your taste" in m.value for m in at.markdown)  # profile setup page

    at.text_input(key="setup_name").input("New User")
    at.multiselect(key="setup_genres").set_value(["r&b", "soul"])
    submit(at, "Save and start listening")
    assert not at.exception
    assert [t.label for t in at.tabs] == ["🎶 Discover", "🧠 How the AI decided", "👤 My profile"]
    assert users.get_profile("newuser").genres == ["r&b", "soul"]


def test_mismatched_passwords_are_caught(users):
    at = start()
    at.text_input(key="signup_user").input("newuser")
    at.text_input(key="signup_pass").input("secret123")
    at.text_input(key="signup_confirm").input("different")
    submit(at, "Create account")
    assert any("don't match" in e.value for e in at.error)


def test_wrong_password_shows_an_error(users):
    users.create_user("alice", "secret123")
    at = start()
    at.text_input(key="login_user").input("alice")
    at.text_input(key="login_pass").input("wrong")
    submit(at, "Log in")
    assert any("Wrong username or password" in e.value for e in at.error)


def test_quick_mix_builds_an_explained_playlist(users):
    """Log in, turn the AI critic off, and run Quick mix: the real agent, fully offline."""
    users.create_user("alice", "secret123")
    users.save_profile("alice", TasteProfile(display_name="Alice", genres=["r&b", "soul"], moods=["chill"],
                                             energy=0.4, avoid_artists=["Khalid"]))
    at = start()
    at.text_input(key="login_user").input("alice")
    at.text_input(key="login_pass").input("secret123")
    submit(at, "Log in")
    at.toggle(key="ai_review").set_value(False)
    at.slider(key="num_songs").set_value(5)
    submit(at, "🎲 Quick mix from my profile")
    assert not at.exception

    result = at.session_state["result"]
    assert len(result.playlist) == 5
    assert all(item.explanation for item in result.playlist)
    assert not any("khalid" in item.song.artist.lower() for item in result.playlist)  # "never play" respected
    assert any("Your playlist" in h.value for h in at.subheader)
