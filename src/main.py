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

from recommender import load_songs, recommend_songs, load_catalog, Recommender


def run_natural_language(query: str) -> None:
    """python src/main.py "chill music for studying, 5 songs" """
    from dotenv import load_dotenv
    from preferences import extract_preferences, ExtractionError

    load_dotenv()  # reads GEMINI_API_KEY from .env
    catalog = load_catalog("data/catalog.csv")

    try:
        request = extract_preferences(query, catalog)
    except ExtractionError as e:
        sys.exit(f"Could not understand the request: {e}")

    p = request.profile
    print(f"\nRequest: {query}")
    print("\n--- What the AI understood ---")
    print(f"Genres:          {p.genres or '(any)'}")
    print(f"Moods:           {p.moods or '(any)'}")
    print(f"Target energy:   {p.target_energy:.2f}")
    print(f"Target valence:  {p.target_valence:.2f}")
    print(f"Likes acoustic:  {p.likes_acoustic}")
    print(f"Songs requested: {request.num_songs}")
    if request.reference_artists:
        print(f"Like artists:    {request.reference_artists}")
    if request.exclude_artists:
        print(f"Exclude artists: {request.exclude_artists}")
    if request.unmatched:
        print(f"Not in catalog:  {request.unmatched}")

    excluded = {a.lower() for a in request.exclude_artists}
    allowed = [s for s in catalog if not any(x in s.artist.lower() for x in excluded)]
    results = Recommender(allowed).recommend_with_scores(p, k=request.num_songs)

    print(f"\n--- Top {len(results)} songs ---")
    for i, (song, score, reasons) in enumerate(results, 1):
        print(f"{i:2}. {song.title} - {song.artist}  [{song.genre}, {song.mood}]  score {score:.2f}")


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
    if len(sys.argv) > 1:
        run_natural_language(" ".join(sys.argv[1:]))
    else:
        main()  # original demo with hardcoded profiles
