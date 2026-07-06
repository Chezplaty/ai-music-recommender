"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


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
    main()
