import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple


@dataclass
class Song:
    """A catalog song and its audio features (all features are 0.0-1.0 except tempo)."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Optional metadata, present in data/catalog.csv but not data/songs.csv
    album: str = ""
    popularity: int = 0
    spotify_track_id: str = ""
    source: str = ""


@dataclass
class UserProfile:
    """A user's taste preferences, in the shape score_song() expects.

    Lists allow several acceptable genres/moods (a song matches if it is in the list).
    This is also the structured output the LLM preference extractor will fill in.
    """
    genres: List[str]
    moods: List[str]
    target_energy: float
    target_valence: float = 0.5
    likes_acoustic: bool = False

    def to_prefs(self) -> Dict:
        """Convert to the dict format used by score_song() / recommend_songs()."""
        return {
            "genre": self.genres,
            "mood": self.moods,
            "energy": self.target_energy,
            "valence": self.target_valence,
            "likes_acoustic": self.likes_acoustic,
        }


class Recommender:
    """OOP wrapper around score_song(), so both interfaces share one scoring formula."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend_with_scores(self, user: UserProfile, k: int = 5) -> List[Tuple[Song, float, List[str]]]:
        """Top k songs as (song, score, reasons), highest score first."""
        prefs = user.to_prefs()
        scored = [(song, *score_song(prefs, asdict(song))) for song in self.songs]
        scored.sort(key=lambda rec: rec[1], reverse=True)
        return scored[:k]

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        return [song for song, _, _ in self.recommend_with_scores(user, k)]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        score, reasons = score_song(user.to_prefs(), asdict(song))
        return f"Score {score:.2f}: " + "; ".join(reasons)


def load_songs(csv_path: str) -> List[Dict]:
    """Reads a songs CSV and returns each row as a dict with numeric fields converted."""
    int_fields = {"id", "tempo_bpm", "popularity"}
    float_fields = {"energy", "valence", "danceability", "acousticness"}

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for name in int_fields & row.keys():
                row[name] = int(row[name])
            for name in float_fields & row.keys():
                row[name] = float(row[name])
            songs.append(row)
    return songs


def load_catalog(csv_path: str) -> List[Song]:
    """Reads a songs CSV into Song objects."""
    return [Song(**row) for row in load_songs(csv_path)]

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Returns a weighted similarity score and the reasons behind it for one song against user_prefs."""
    genre_sim = 1.0 if song["genre"] in user_prefs["genre"] else 0.0
    mood_sim = 1.0 if song["mood"] in user_prefs["mood"] else 0.0

    target_energy = user_prefs["energy"]
    target_valence = user_prefs.get("valence", 0.5)
    target_acousticness = 1.0 if user_prefs.get("likes_acoustic", False) else 0.0

    energy_sim = 1 - abs(target_energy - song["energy"])
    valence_sim = 1 - abs(target_valence - song["valence"])
    acoustic_sim = 1 - abs(target_acousticness - song["acousticness"])

    score = (
        0.30 * genre_sim
        + 0.15 * mood_sim
        + 0.35 * energy_sim
        + 0.10 * valence_sim
        + 0.10 * acoustic_sim
    )

    reasons = []
    if genre_sim == 1.0:
        reasons.append(f"Matches favorite genre '{song['genre']}'")
    else:
        reasons.append(f"Different genre ('{song['genre']}' vs favorites {user_prefs['genre']})")

    if mood_sim == 1.0:
        reasons.append(f"Matches favorite mood '{song['mood']}'")
    else:
        reasons.append(f"Different mood ('{song['mood']}' vs favorites {user_prefs['mood']})")

    reasons.append(f"Energy similarity {energy_sim:.2f} (song {song['energy']}, target {target_energy})")
    reasons.append(f"Valence similarity {valence_sim:.2f} (song {song['valence']}, target {target_valence})")
    reasons.append(f"Acousticness similarity {acoustic_sim:.2f} (song {song['acousticness']}, target {target_acousticness})")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Scores every song against user_prefs and returns the top k, highest score first."""
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = "\n".join(reasons)
        scored.append((song, score, explanation))

    scored.sort(key=lambda rec: rec[1], reverse=True)
    return scored[:k]
