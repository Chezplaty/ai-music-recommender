"""
Build data/catalog.csv by merging two public Spotify audio-feature datasets.

Sources
-------
1. TidyTuesday Spotify Songs (2020-01-21), ~32.8k tracks, 6 genres + 24 subgenres.
   Mostly mainstream/Western music; used for pop, hip hop, r&b, soul, rock, edm, latin.
   https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-01-21
   Raw: https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-01-21/spotify_songs.csv

2. Spotify Tracks Dataset (maharshipandya, Hugging Face), 114k rows, 114 genres.
   Its pop/hip-hop genres skew heavily toward Indian/Punjabi tracks, so it is only
   used for the genres TidyTuesday lacks (jazz, classical, folk, metal, ...).
   https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
   Raw: https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv

Usage
-----
    python scripts/build_catalog.py --tidytuesday <spotify_songs.csv> --hf <dataset.csv> --out data/catalog.csv

Sampling steps (deterministic with SEED)
----------------------------------------
  1. Map each source genre/subgenre to one of 25 catalog genres (GENRES below).
  2. Drop low-quality rows: missing title/artist, popularity < MIN_POPULARITY,
     speechiness > MAX_SPEECHINESS (spoken word / comedy), remix titles.
  3. For each catalog genre, shuffle its candidates and take PER_GENRE songs,
     skipping duplicates (same Spotify track_id, or same normalized artist+title
     in EITHER source) and artists that already have MAX_PER_ARTIST songs.
  4. Derive a `mood` label from energy/valence. Neither source has mood labels,
     so this is a heuristic, NOT ground truth. See derive_mood().
"""

import argparse
import csv
import random
import re
import sys
from collections import Counter

SEED = 42
PER_GENRE = 12
MIN_POPULARITY = 40
MAX_SPEECHINESS = 0.66
MAX_PER_ARTIST = 2

TT = "tidytuesday"
HF = "spotify-tracks-hf"

# (catalog genre, source, raw genre keys)
#   TidyTuesday keys are (playlist_genre, playlist_subgenre)
#   Hugging Face keys are track_genre
GENRES = [
    ("pop", TT, {("pop", "dance pop"), ("pop", "post-teen pop"), ("pop", "electropop")}),
    ("indie pop", TT, {("pop", "indie poptimism")}),
    ("hip hop", TT, {("rap", "hip hop"), ("rap", "southern hip hop"), ("rap", "gangster rap"), ("rap", "trap")}),
    ("r&b", TT, {("r&b", "urban contemporary"), ("r&b", "new jack swing"), ("r&b", "hip pop")}),
    ("soul", TT, {("r&b", "neo soul")}),
    ("rock", TT, {("rock", "album rock"), ("rock", "classic rock"), ("rock", "permanent wave"), ("rock", "hard rock")}),
    ("edm", TT, {("edm", "big room"), ("edm", "electro house"), ("edm", "pop edm"), ("edm", "progressive electro house")}),
    ("latin", TT, {("latin", "latin pop"), ("latin", "tropical"), ("latin", "latin hip hop"), ("latin", "reggaeton")}),
    ("alt rock", HF, {"alt-rock"}),
    ("punk", HF, {"punk"}),
    ("metal", HF, {"metal"}),
    ("synthpop", HF, {"synth-pop"}),
    ("k-pop", HF, {"k-pop"}),
    ("funk", HF, {"funk"}),
    ("jazz", HF, {"jazz"}),
    ("blues", HF, {"blues"}),
    ("country", HF, {"country"}),
    ("folk", HF, {"folk"}),
    ("acoustic", HF, {"acoustic"}),
    ("classical", HF, {"classical"}),
    ("ambient", HF, {"ambient"}),
    ("chill", HF, {"chill"}),
    ("house", HF, {"house"}),
    ("techno", HF, {"techno"}),
    ("reggae", HF, {"reggae"}),
]

OUTPUT_FIELDS = [
    "id", "title", "artist", "genre", "mood", "energy", "tempo_bpm",
    "valence", "danceability", "acousticness",
    "album", "popularity", "spotify_track_id", "source",
]


def derive_mood(energy: float, valence: float, danceability: float) -> str:
    """Map audio features to a mood using energy x valence quadrants
    (Russell's circumplex model of affect), with sub-labels at the extremes."""
    if energy >= 0.6:
        if valence >= 0.5:
            return "euphoric" if energy >= 0.8 and danceability >= 0.7 else "happy"
        return "angry" if energy >= 0.85 and valence < 0.3 else "intense"
    if valence >= 0.5:
        return "chill" if energy < 0.4 else "relaxed"
    return "sad" if valence < 0.25 else "melancholic"


def normalize(text: str) -> str:
    """Lowercase and strip ' - Remastered 2011', '(feat. X)' etc. for dedupe."""
    text = text.lower()
    text = re.sub(r"\s*[\(\[].*?[\)\]]", "", text)
    text = re.sub(r"\s+-\s+.*$", "", text)
    return text.strip()


def read_csv(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_tidytuesday(path: str) -> list:
    return [{
        "source": TT,
        "genre_key": (r["playlist_genre"], r["playlist_subgenre"]),
        "track_id": r["track_id"],
        "title": r["track_name"],
        "artist": r["track_artist"],
        "album": r["track_album_name"],
        "popularity": r["track_popularity"],
        **{k: r[k] for k in ("energy", "valence", "danceability", "acousticness", "speechiness", "tempo")},
    } for r in read_csv(path)]


def load_hf(path: str) -> list:
    return [{
        "source": HF,
        "genre_key": r["track_genre"],
        "track_id": r["track_id"],
        "title": r["track_name"],
        "artist": r["artists"].replace(";", ", "),
        "album": r["album_name"],
        "popularity": r["popularity"],
        **{k: r[k] for k in ("energy", "valence", "danceability", "acousticness", "speechiness", "tempo")},
    } for r in read_csv(path)]


def is_clean(r: dict) -> bool:
    return (
        bool(r["title"]) and bool(r["artist"])
        and int(r["popularity"]) >= MIN_POPULARITY
        and float(r["speechiness"]) <= MAX_SPEECHINESS
        and "remix" not in r["title"].lower()
    )


def main(tt_path: str, hf_path: str, out_path: str) -> None:
    rows = [r for r in load_tidytuesday(tt_path) + load_hf(hf_path) if is_clean(r)]
    rng = random.Random(SEED)

    seen_ids, seen_songs = set(), set()
    artist_counts = Counter()
    out_rows = []

    for catalog_genre, source, keys in GENRES:
        candidates = [r for r in rows if r["source"] == source and r["genre_key"] in keys]
        rng.shuffle(candidates)
        picked = 0
        for r in candidates:
            if picked >= PER_GENRE:
                break
            primary_artist = r["artist"].split(",")[0].strip()
            song_key = (normalize(primary_artist), normalize(r["title"]))
            if r["track_id"] in seen_ids or song_key in seen_songs:
                continue
            if artist_counts[primary_artist] >= MAX_PER_ARTIST:
                continue
            seen_ids.add(r["track_id"])
            seen_songs.add(song_key)
            artist_counts[primary_artist] += 1
            picked += 1

            energy, valence, dance = float(r["energy"]), float(r["valence"]), float(r["danceability"])
            out_rows.append({
                "id": len(out_rows) + 1,
                "title": r["title"],
                "artist": r["artist"],
                "genre": catalog_genre,
                "mood": derive_mood(energy, valence, dance),
                "energy": round(energy, 3),
                "tempo_bpm": round(float(r["tempo"])),
                "valence": round(valence, 3),
                "danceability": round(dance, 3),
                "acousticness": round(float(r["acousticness"]), 3),
                "album": r["album"],
                "popularity": int(r["popularity"]),
                "spotify_track_id": r["track_id"],
                "source": r["source"],
            })
        if picked < PER_GENRE:
            print(f"WARNING: genre '{catalog_genre}' only has {picked} songs", file=sys.stderr)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} songs to {out_path}")
    print("Genres:", len({r["genre"] for r in out_rows}))
    print("Sources:", dict(Counter(r["source"] for r in out_rows)))
    print("Moods:", dict(Counter(r["mood"] for r in out_rows).most_common()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tidytuesday", required=True, help="path to TidyTuesday spotify_songs.csv")
    parser.add_argument("--hf", required=True, help="path to Hugging Face spotify-tracks dataset.csv")
    parser.add_argument("--out", default="data/catalog.csv")
    args = parser.parse_args()
    main(args.tidytuesday, args.hf, args.out)
