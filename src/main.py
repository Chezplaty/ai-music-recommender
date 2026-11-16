"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import logging
import sys
import time

from recommender import load_songs, recommend_songs, load_catalog


MODES = ("agent", "agent-rules", "rag", "baseline")


def print_understood(query, request) -> None:
    p = request.profile
    print(f"\nRequest: {query}")
    print("\n--- What the AI understood ---")
    print(f"Genres:          {p.genres or '(any)'}")
    print(f"Moods:           {p.moods or '(any)'}")
    print(f"Target energy:   {p.target_energy:.2f}")
    print(f"Target valence:  {p.target_valence:.2f}")
    print(f"Acousticness:    {p.target_acousticness:.2f}")
    print(f"Songs requested: {request.num_songs}")
    if request.reference_artists:
        print(f"Like artists:    {request.reference_artists}")
    if request.exclude_artists:
        print(f"Exclude artists: {request.exclude_artists}")
    if request.discover_new_artists:
        print("Discovery:       wants new artists")
    if request.unmatched:
        print(f"Not in catalog:  {request.unmatched}")


def print_agent_result(result) -> None:
    current = None
    print("\n--- What the agent did ---")
    for entry in result.trace:
        if entry["step"] in ("analyze",):
            continue
        if entry["step"] != current:
            current = entry["step"]
            print(f"[{current}]")
        print(f"   {entry['detail']}")

    print(f"\n--- Playlist ({len(result.playlist)} songs) ---")
    for i, item in enumerate(result.playlist, 1):
        s = item.song
        match = f", match {item.similarity:.2f}" if item.similarity is not None else ""
        print(f"{i:2}. {s.title} - {s.artist}  [{s.genre}, {s.mood}]")
        print(f"      score {item.score:.2f} ({item.confidence} confidence{match}): "
              + "; ".join(item.reasons[:2]))
    if result.notes:
        print("\n--- Notes ---")
        for note in result.notes:
            print(f" * {note}")


def run_natural_language(query: str, mode: str = "agent") -> None:
    """python src/main.py "chill music for studying, 5 songs"         (agent, default)
    python src/main.py --mode agent-rules "..." (agent with rule checks only, no Gemini critic)
    python src/main.py --mode rag "..."        (retrieve, then rank once; no critic)
    python src/main.py --mode baseline "..."   (original recommender: score the whole catalog)
    """
    from dotenv import load_dotenv
    from preferences import extract_preferences, ExtractionError

    load_dotenv()  # reads GEMINI_API_KEY from .env
    catalog = load_catalog("data/catalog.csv")

    retriever = None
    if mode in ("agent", "agent-rules", "rag"):
        from rag.retrieval import build_retriever
        print("Loading music knowledge base (first run builds the index)...", flush=True)
        retriever = build_retriever(catalog)

    if mode == "agent":
        print("Running the agent (Gemini interprets, then reviews each attempt; this can take a few minutes)...", flush=True)
    else:
        print("Asking Gemini to interpret your request...", flush=True)
    start = time.time()
    try:
        if mode.startswith("agent"):
            from agent.workflow import run_agent
            result = run_agent(query, catalog, retriever, ai_review=(mode == "agent"))
            request = result.request
        else:
            request = extract_preferences(query, catalog)
    except ExtractionError as e:
        sys.exit(f"Could not understand the request: {e}")
    print(f"(took {time.time() - start:.1f}s)")

    print_understood(query, request)
    if mode.startswith("agent"):
        print_agent_result(result)
        return

    from rag.pipeline import recommend
    playlist = recommend(request, catalog, retriever=retriever, query=query)

    if playlist.mode == "rag":
        print("\n--- Retrieval (RAG) ---")
        print(f"Search text:     {playlist.search_text}")
        print(f"Candidates:      {len(playlist.retrieved)} of {len(catalog)} songs retrieved, then ranked by the scorer")
        print("Closest matches:")
        for r in playlist.retrieved[:5]:
            print(f"  {r.similarity:.2f}  {r.song.title} - {r.song.artist}  [{r.song.genre}, {r.song.mood}]")
    else:
        print("\n(RAG disabled: scoring the whole catalog)")

    sims = playlist.similarity_by_id
    print(f"\n--- Top {len(playlist.results)} songs ---")
    for i, (song, score, reasons) in enumerate(playlist.results, 1):
        sim = f"  match {sims[song.id]:.2f}" if song.id in sims else ""
        print(f"{i:2}. {song.title} - {song.artist}  [{song.genre}, {song.mood}]  score {score:.2f}{sim}")


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs from data/songs.csv")

    profiles = {
        "Chill lofi listener": {
            "genre": ["lofi"],
            "mood": ["chill"],
            "energy": 0.45,
            "likes_acoustic": True,
        },
        # Edge case: maxed-out energy (1.0) plus likes_acoustic=True, which
        # contradicts typical high-energy/electronic songs (usually low
        # acousticness) - tests how the scorer trades off conflicting signals.
        "High-energy upbeat listener (adversarial)": {
            "genre": ["pop", "house", "edm"],  # "edm" doesn't exist in the dataset
            "mood": ["euphoric", "hyped"],  # "hyped" doesn't exist in the dataset
            "energy": 1.0,
            "likes_acoustic": True,
        },
        # Edge case: wants "sad" metal/rock but sets a low target energy
        # (0.1), even though metal/rock songs in the dataset are all
        # high-energy (0.9+) - genre match and energy match directly conflict.
        "Sad metal/rock listener (adversarial)": {
            "genre": ["metal", "rock"],
            "mood": ["sad", "heartbroken"],  # "heartbroken" doesn't exist in the dataset
            "energy": 0.1,
            "likes_acoustic": True,
        },
    }

    for profile_name, user_prefs in profiles.items():
        print(f"\n=== {profile_name} ===")
        recommendations = recommend_songs(user_prefs, songs, k=5)

        print("\nTop recommendations:\n")
        for rec in recommendations:
            # You decide the structure of each returned item.
            # A common pattern is: (song, score, explanation)
            song, score, explanation = rec
            title_line = f"{song['title']} - Score: {score:.2f}"
            print(title_line)
            print("-" * len(title_line))
            print(f"Because: {explanation}")
            print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = sys.argv[1:]
    mode = "agent"
    if "--mode" in args:
        i = args.index("--mode")
        if i + 1 >= len(args) or args[i + 1] not in MODES:
            sys.exit(f"--mode must be one of {', '.join(MODES)}")
        mode = args[i + 1]
        del args[i:i + 2]
    if args:
        run_natural_language(" ".join(args), mode=mode)
    else:
        main()  # original demo with hardcoded profiles
