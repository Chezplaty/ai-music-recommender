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

    # Starter example profile
    user_prefs = {"genre": ["lofi"], "mood": ["chill"], "energy": 0.45, "likes_acoustic": True}

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
