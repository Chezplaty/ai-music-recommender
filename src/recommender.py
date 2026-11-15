import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
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

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Reads data/songs.csv and returns each row as a dict with numeric fields converted."""
    int_fields = {"id", "tempo_bpm"}
    float_fields = {"energy", "valence", "danceability", "acousticness"}

    songs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            songs.append(row)
    return songs

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
