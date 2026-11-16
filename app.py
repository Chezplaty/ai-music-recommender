"""
Streamlit app for the AI Music Discovery Agent.

    streamlit run app.py

Flow: log in / sign up -> create a taste profile -> discover music.
  Discover            type a request (or make a quick mix from your profile); the agent
                      shows each step live, then the playlist with explanations
  How the AI decided  what Gemini understood, every attempt's rule checks, the AI
                      critic's verdict, and the revisions it made
  My profile          edit your saved taste profile
"""

import logging
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.WARNING)

from accounts import AccountError, TasteProfile, UserStore, parse_artist_list  # noqa: E402
from preferences import ExtractionError, build_vocabulary, extract_preferences  # noqa: E402
from recommender import load_catalog  # noqa: E402

st.set_page_config(page_title="Music Discovery Agent", page_icon="🎧", layout="wide")

EXAMPLES = [
    "Studying for finals: the vibe of Frank Ocean's Blonde, but more upbeat and not distracting",
    "Late-night drive. I like The Weeknd, but I want to discover new artists",
    "Loud, angry music for the gym, no country",
    "Rainy Sunday morning, soft acoustic songs",
]
STEP_LABELS = {
    "analyze": "🧠 Understood your request",
    "plan": "🗺️ Planned the search",
    "retrieve": "🔎 Searched the music library",
    "generate": "📊 Ranked the candidates",
    "evaluate": "✅ Checked the playlist",
    "ai_review": "🧑‍⚖️ AI critic",
    "revise": "🔁 Revised the plan",
    "finalize": "🏁 Picked the best attempt",
    "explain": "💬 Wrote the explanations",
}
CONFIDENCE_COLORS = {"high": "green", "medium": "orange", "low": "red"}

st.markdown("""
<style>
  .block-container { padding-top: 2rem; max-width: 1100px; }
  .rank { font-size: 1.6rem; font-weight: 700; opacity: .55; text-align: center; padding-top: .2rem; }
  .song-title { font-size: 1.1rem; font-weight: 650; margin-bottom: 0; }
  .song-artist { opacity: .75; margin-top: 0; }
  .hero { font-size: 2.2rem; font-weight: 750; margin-bottom: 0; }
  .muted { opacity: .7; }
</style>
""", unsafe_allow_html=True)


# --- shared resources ---------------------------------------------------------------

@st.cache_resource
def get_catalog():
    return load_catalog(str(ROOT / "data" / "catalog.csv"))


@st.cache_resource(show_spinner="Loading the music library (first time only)...")
def get_retriever():
    from rag.retrieval import build_retriever
    return build_retriever(get_catalog())


@st.cache_resource
def get_vocab():
    return build_vocabulary(get_catalog())


def store() -> UserStore:
    return UserStore()


def logout():
    for key in ("user", "profile", "result", "query", "run_error"):
        st.session_state.pop(key, None)


# --- 1. log in / sign up ------------------------------------------------------------------

def render_auth():
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<p class="hero">🎧 Music Discovery Agent</p>', unsafe_allow_html=True)
        st.markdown(
            "Describe what you want to hear, in your own words. An AI agent searches a 300-song "
            "library, ranks the matches, **checks its own work**, and revises the playlist until it fits. "
            "Then it tells you why each song made the cut."
        )
        st.markdown("- 🔎 Semantic search over song descriptions (ChromaDB)\n"
                    "- 📊 Transparent scoring on genre, mood, energy, positivity and acousticness\n"
                    "- 🧑‍⚖️ A Gemini critic that reviews and revises each attempt\n"
                    "- 🛡️ Guardrails: only real catalog songs, no duplicates, your exclusions respected")
    with right:
        login_tab, signup_tab = st.tabs(["Log in", "Create account"])
        with login_tab:
            with st.form("login"):
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                if st.form_submit_button("Log in", type="primary", use_container_width=True):
                    try:
                        user = store().authenticate(username, password)
                        st.session_state.user = user
                        st.session_state.profile = store().get_profile(user)
                        st.rerun()
                    except AccountError as e:
                        st.error(str(e))
        with signup_tab:
            with st.form("signup"):
                username = st.text_input("Choose a username", key="signup_user",
                                         help="3-30 characters: letters, numbers, underscores")
                password = st.text_input("Choose a password", type="password", key="signup_pass")
                confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
                if st.form_submit_button("Create account", type="primary", use_container_width=True):
                    if password != confirm:
                        st.error("Passwords don't match.")
                    else:
                        try:
                            st.session_state.user = store().create_user(username, password)
                            st.session_state.profile = None
                            st.rerun()
                        except AccountError as e:
                            st.error(str(e))


# --- 2. taste profile -----------------------------------------------------------------------

def profile_form(existing: TasteProfile, submit_label: str, form_key: str):
    """Returns a TasteProfile when submitted, else None."""
    vocab = get_vocab()
    with st.form(form_key):
        name = st.text_input("What should we call you?", value=existing.display_name, key=f"{form_key}_name")
        c1, c2 = st.columns(2)
        genres = c1.multiselect("Favorite genres", vocab["genres"], default=existing.genres,
                                key=f"{form_key}_genres", help="Pick a few, or none for 'anything goes'")
        moods = c2.multiselect("Favorite moods", vocab["moods"], default=existing.moods, key=f"{form_key}_moods")

        st.markdown("**Your usual sound**")
        s1, s2, s3 = st.columns(3)
        energy = s1.slider("Energy", 0.0, 1.0, existing.energy, 0.05, key=f"{form_key}_energy",
                           help="0 = calm and quiet, 1 = loud and intense")
        valence = s2.slider("Positivity", 0.0, 1.0, existing.valence, 0.05, key=f"{form_key}_valence",
                            help="0 = dark or sad, 1 = bright and happy")
        acoustic = s3.slider("Acoustic", 0.0, 1.0, existing.acousticness, 0.05, key=f"{form_key}_acoustic",
                             help="0 = electronic and produced, 1 = unplugged")

        a1, a2 = st.columns(2)
        favorites = a1.text_area("Artists you love", value=", ".join(existing.favorite_artists),
                                 key=f"{form_key}_fav", placeholder="Frank Ocean, SZA, ...", height=80)
        avoid = a2.text_area("Artists to never play", value=", ".join(existing.avoid_artists),
                             key=f"{form_key}_avoid", placeholder="Comma-separated", height=80)

        if st.form_submit_button(submit_label, type="primary"):
            if not name.strip():
                st.error("Please add a name.")
                return None
            return TasteProfile(display_name=name.strip(), genres=genres, moods=moods, energy=energy,
                                valence=valence, acousticness=acoustic,
                                favorite_artists=parse_artist_list(favorites), avoid_artists=parse_artist_list(avoid))
    return None


def render_profile_setup():
    st.markdown('<p class="hero">Welcome! Let\'s get to know your taste 🎶</p>', unsafe_allow_html=True)
    st.markdown('<p class="muted">This becomes the starting point for every playlist. '
                'You can change it any time.</p>', unsafe_allow_html=True)
    profile = profile_form(TasteProfile(display_name=st.session_state.user), "Save and start listening",
                           "setup")
    if profile:
        store().save_profile(st.session_state.user, profile)
        st.session_state.profile = profile
        st.rerun()
    st.button("Log out", on_click=logout)


# --- 3. the app -----------------------------------------------------------------------------

def render_sidebar(profile: TasteProfile):
    with st.sidebar:
        st.markdown(f"### 🎧 Hi, {profile.display_name}")
        if profile.genres:
            st.caption("Your genres: " + ", ".join(profile.genres))
        if profile.moods:
            st.caption("Your moods: " + ", ".join(profile.moods))
        st.divider()
        st.toggle("AI critic (Gemini)", value=True, key="ai_review",
                  help="On: Gemini reviews each attempt and writes the revisions. "
                       "Off: rule checks only (faster, works when Gemini is down).")
        if not os.environ.get("GEMINI_API_KEY"):
            st.warning("No GEMINI_API_KEY found in .env. Typed requests need it; "
                       "Quick mix with the AI critic off works without it.")
        st.divider()
        st.button("Log out", on_click=logout, use_container_width=True)


def live_line(entries) -> str:
    """One friendly progress line per finished agent step."""
    step = entries[0]["step"]
    label = STEP_LABELS.get(step, step)
    if step == "evaluate":
        checks = [e for e in entries if e["step"] == "evaluate"]
        passed = sum(e["detail"].startswith("PASS") for e in checks)
        line = f"{label}: {passed}/{len(checks)} rule checks passed"
        reviews = [e["detail"] for e in entries if e["step"] == "ai_review"]
        if reviews:
            line += f"  \n{STEP_LABELS['ai_review']}: {reviews[0]}"
        return line
    if step == "revise":
        return f"{label}: " + "; ".join(e["detail"].split(": ", 1)[-1] for e in entries)
    return f"{label}: {entries[-1]['detail']}"


def run_and_show_progress(query, num_songs, request=None):
    from agent.workflow import iter_agent

    profile: TasteProfile = st.session_state.profile
    catalog, retriever = get_catalog(), get_retriever()

    def extractor(q, cat):
        parsed = extract_preferences(q, cat, listener_profile=profile.to_prompt_text())
        parsed.num_songs = num_songs  # the slider is the explicit choice
        # The profile's "never play" list is a hard rule, even if Gemini didn't carry it over.
        parsed.exclude_artists += [a for a in profile.avoid_artists if a not in parsed.exclude_artists]
        return parsed

    st.session_state.pop("result", None)
    st.session_state.pop("run_error", None)
    with st.status("Building your playlist... (Gemini can take a minute per step)", expanded=True) as status:
        try:
            result = None
            for kind, payload in iter_agent(query, catalog, retriever, extractor=extractor, request=request,
                                            ai_review=st.session_state.get("ai_review", True)):
                if kind == "step" and payload:
                    st.markdown(live_line(payload))
                elif kind == "result":
                    result = payload
            st.session_state.result = result
            status.update(label=f"Playlist ready: {len(result.playlist)} songs", state="complete", expanded=False)
        except ExtractionError as e:
            st.session_state.run_error = (
                f"Gemini couldn't interpret the request: {e}\n\n"
                "Gemini is often overloaded. Try again in a minute, or use **Quick mix from my profile** "
                "with the AI critic turned off in the sidebar.")
            status.update(label="Couldn't build the playlist", state="error")


def song_card(rank: int, item):
    s = item.song
    with st.container(border=True):
        c_rank, c_main, c_side = st.columns([0.5, 6, 1.6], vertical_alignment="center")
        c_rank.markdown(f'<div class="rank">{rank}</div>', unsafe_allow_html=True)
        with c_main:
            st.markdown(f'<p class="song-title">{s.title}</p><p class="song-artist">{s.artist}'
                        + (f" · <i>{s.album}</i>" if s.album else "") + "</p>", unsafe_allow_html=True)
            b = st.columns([1, 1, 1.4, 3])
            with b[0]:
                st.badge(s.genre, color="violet")
            with b[1]:
                st.badge(s.mood, color="blue")
            with b[2]:
                st.badge(f"{item.confidence} confidence", color=CONFIDENCE_COLORS.get(item.confidence, "gray"))
        with c_side:
            st.metric("Score", f"{item.score:.2f}", label_visibility="visible")
            if s.spotify_track_id:
                st.link_button("▶ Spotify", f"https://open.spotify.com/track/{s.spotify_track_id}",
                               use_container_width=True)
        st.markdown(item.explanation)
        with st.expander("Evidence"):
            for fact in item.evidence:
                icon = "✅" if fact.supports else "⚠️" if fact.supports is False else "•"
                st.markdown(f"{icon} **{fact.id}**: {fact.text}")


def render_discover():
    profile: TasteProfile = st.session_state.profile

    def use_example():
        if st.session_state.get("example"):
            st.session_state.query = st.session_state.example

    st.text_area("What do you want to listen to?", key="query", height=90,
                 placeholder="e.g. upbeat songs for a morning run, something like Dua Lipa but less pop")
    st.pills("Need an idea?", EXAMPLES, key="example", on_change=use_example)

    c1, c2, c3 = st.columns([1.3, 1, 1], vertical_alignment="bottom")
    num_songs = c1.slider("Songs", 5, 20, 10, key="num_songs")
    generate = c2.button("✨ Generate playlist", type="primary", use_container_width=True)
    quick = c3.button("🎲 Quick mix from my profile", use_container_width=True,
                      help="Skips request interpretation and builds a mix straight from your saved profile")

    if generate:
        query = (st.session_state.get("query") or "").strip()
        if not query:
            st.warning("Type a request first, or try Quick mix.")
        else:
            run_and_show_progress(query, num_songs)
    if quick:
        run_and_show_progress("A mix based on my saved profile", num_songs, request=profile.to_request(num_songs))

    if st.session_state.get("run_error"):
        st.error(st.session_state.run_error)
    result = st.session_state.get("result")
    if not result:
        return

    st.subheader(f"Your playlist · {len(result.playlist)} songs")
    if result.summary:
        st.markdown(f'<p class="muted">{result.summary}</p>', unsafe_allow_html=True)
    if any(a.review and a.review.approved for a in result.attempts):
        st.success("🧑‍⚖️ The AI critic approved this playlist. See **How the AI decided** for its reasoning.")
    for note in result.notes:
        st.info(note)
    for rank, item in enumerate(result.playlist, 1):
        song_card(rank, item)


def render_how_it_decided():
    result = st.session_state.get("result")
    if not result:
        st.info("Generate a playlist first. This tab will show every decision the agent made.")
        return
    req, p = result.request, result.request.profile

    st.markdown("#### 1 · What the AI understood")
    c1, c2 = st.columns(2)
    c1.markdown("**Genres:** " + (", ".join(p.genres) or "any"))
    c1.markdown("**Moods:** " + (", ".join(p.moods) or "any"))
    if req.reference_artists:
        c1.markdown("**Like:** " + ", ".join(req.reference_artists)
                    + (" (but wants new artists)" if req.discover_new_artists else ""))
    if req.exclude_artists:
        c1.markdown("**Never:** " + ", ".join(req.exclude_artists))
    for label, value in (("Energy", p.target_energy), ("Positivity", p.target_valence),
                         ("Acoustic", p.target_acousticness)):
        c2.progress(value, text=f"{label}: {value:.2f}")
    if req.search_description:
        st.caption(f"Sound searched for: “{req.search_description}”")

    st.markdown("#### 2 · Attempts")
    revisions = [t["detail"] for t in result.trace if t["step"] == "revise"]
    for attempt in result.attempts:
        chosen = attempt.number == result.chosen_attempt
        ev = attempt.evaluation
        title = (f"Attempt {attempt.number} · {len(ev.checks) - len(ev.failed)}/{len(ev.checks)} rule checks"
                 + (" · AI approved" if attempt.review and attempt.review.approved else "")
                 + (" · ⭐ returned" if chosen else ""))
        with st.expander(title, expanded=chosen):
            s = attempt.strategy
            st.caption(f"Searched {s.n_candidates} songs for “{s.search_text.splitlines()[0]}”"
                       + (" + a search inside each genre" if s.genre_targeted else ""))
            st.dataframe([{"check": c.name, "result": "✅ pass" if c.passed else "❌ fail",
                           "type": "guardrail" if c.hard else "quality", "detail": c.detail}
                          for c in ev.checks], hide_index=True, use_container_width=True)
            r = attempt.review
            if r is not None:
                if r.error:
                    st.warning("AI critic unavailable for this attempt; rule checks were used instead.")
                else:
                    verdict = "✅ Approved" if r.approved else "❌ Rejected"
                    st.markdown(f"**🧑‍⚖️ AI critic: {verdict}.** {r.reasoning}")
                    for issue in r.issues:
                        st.markdown(f"- {issue}")
            mine = [d for d in revisions if d.startswith(f"revision {attempt.number} ")]
            if mine:
                st.markdown("**🔁 Changes for the next attempt:**")
                for d in mine:
                    st.markdown(f"- {d.split(': ', 1)[-1]}")
            st.markdown("**Songs:** " + ", ".join(f"{i.song.title} ({i.score:.2f})" for i in attempt.playlist))

    with st.expander("Full agent trace"):
        st.dataframe(result.trace, hide_index=True, use_container_width=True)


def render_my_profile():
    profile = profile_form(st.session_state.profile, "Save changes", "edit")
    if profile:
        store().save_profile(st.session_state.user, profile)
        st.session_state.profile = profile
        st.success("Saved. Your next playlist will use the new profile.")


def render_app():
    profile: TasteProfile = st.session_state.profile
    render_sidebar(profile)
    st.markdown('<p class="hero">🎧 Music Discovery Agent</p>', unsafe_allow_html=True)
    discover, how, me = st.tabs(["🎶 Discover", "🧠 How the AI decided", "👤 My profile"])
    with discover:
        render_discover()
    with how:
        render_how_it_decided()
    with me:
        render_my_profile()


def main():
    if not st.session_state.get("user"):
        render_auth()
    elif st.session_state.get("profile") is None:
        render_profile_setup()
    else:
        render_app()


main()
