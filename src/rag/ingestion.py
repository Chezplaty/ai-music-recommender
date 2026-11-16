"""
Build the music knowledge base: one rich text document per catalog song, stored in ChromaDB.

Embedding just "title + artist" would make search nearly useless (the model knows
little about most tracks). Instead each song becomes a short catalog entry that
combines:
  - real metadata from data/catalog.csv (title, artist, album, genre, mood)
  - plain-English descriptions of its audio features (energy 0.85 -> "energetic, upbeat")
  - genre-level knowledge from data/music_knowledge.json (style, characteristics, related genres)

Nothing here describes a specific song beyond what the catalog numbers say, so the
documents contain no invented facts about individual tracks.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

from recommender import Song
from preferences import MOOD_DEFINITIONS
from rag.embeddings import Embedder

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "music_knowledge.json"
COLLECTION_NAME = "songs"
BATCH_SIZE = 64


def load_genre_knowledge(path: Path = KNOWLEDGE_PATH) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["genres"]


# --- audio features -> words --------------------------------------------------
# Thresholds are on Spotify's 0-1 scales. Words are chosen to overlap with how
# people phrase requests ("upbeat", "mellow", "acoustic") so semantic search can match them.

def describe_energy(x: float) -> str:
    if x < 0.3:
        return "very calm, low energy, quiet"
    if x < 0.5:
        return "mellow, low-to-moderate energy"
    if x < 0.7:
        return "moderate energy"
    if x < 0.85:
        return "energetic, upbeat"
    return "very high energy, intense, loud"


def describe_valence(x: float) -> str:
    if x < 0.25:
        return "dark, sad"
    if x < 0.45:
        return "moody, bittersweet"
    if x < 0.65:
        return "emotionally balanced"
    if x < 0.8:
        return "bright, positive"
    return "very happy, joyful, feel-good"


def describe_acousticness(x: float) -> str:
    if x >= 0.7:
        return "acoustic, organic instrumentation"
    if x >= 0.4:
        return "mix of acoustic and electronic sounds"
    return "electronic, heavily produced"


def describe_danceability(x: float) -> str:
    if x >= 0.7:
        return "very danceable, strong groove"
    if x >= 0.45:
        return "moderately danceable"
    return "not dance-oriented"


def describe_tempo(bpm: float) -> str:
    if bpm < 90:
        return f"slow tempo ({bpm:.0f} BPM)"
    if bpm < 120:
        return f"mid tempo ({bpm:.0f} BPM)"
    return f"fast tempo ({bpm:.0f} BPM)"


def song_to_document(song: Song, genre_knowledge: Dict[str, dict]) -> str:
    """The text that gets embedded for one song."""
    genre = genre_knowledge.get(song.genre, {})
    sound = "; ".join([
        describe_energy(song.energy),
        describe_valence(song.valence),
        describe_acousticness(song.acousticness),
        describe_danceability(song.danceability),
        describe_tempo(song.tempo_bpm),
    ])
    lines = [
        f"Song: {song.title}",
        f"Artist: {song.artist}",
        f"Album: {song.album}" if song.album else "",
        f"Genre: {song.genre}" + (f" ({genre['style']})" if genre.get("style") else ""),
        f"Mood: {song.mood} ({MOOD_DEFINITIONS.get(song.mood, '')})",
        f"Sound: {sound}",
        f"Genre characteristics: {genre['characteristics']}" if genre.get("characteristics") else "",
        f"Related genres: {', '.join(genre['related'])}" if genre.get("related") else "",
    ]
    return "\n".join(line for line in lines if line)


def song_metadata(song: Song) -> dict:
    """Stored next to each vector. Lets Chroma filter (e.g. by genre) without re-reading the CSV."""
    return {
        "song_id": song.id,
        "title": song.title,
        "artist": song.artist,
        "album": song.album,
        "genre": song.genre,
        "mood": song.mood,
        "energy": song.energy,
        "valence": song.valence,
        "acousticness": song.acousticness,
        "danceability": song.danceability,
        "tempo_bpm": float(song.tempo_bpm),
    }


def fingerprint(documents: List[str], embedder_name: str) -> str:
    """Changes whenever the catalog, the knowledge file, or the embedding model changes."""
    h = hashlib.sha256(embedder_name.encode())
    for doc in documents:
        h.update(b"\x00" + doc.encode())
    return h.hexdigest()[:16]


def ensure_index(chroma_client, catalog: List[Song], embedder: Embedder,
                 genre_knowledge: Dict[str, dict] = None, collection_name: str = COLLECTION_NAME):
    """Return an up-to-date Chroma collection for `catalog`, (re)building it only if needed.

    Returns (collection, rebuilt: bool).
    """
    genre_knowledge = load_genre_knowledge() if genre_knowledge is None else genre_knowledge
    documents = [song_to_document(s, genre_knowledge) for s in catalog]
    fp = fingerprint(documents, embedder.name)

    existing = {c.name if hasattr(c, "name") else c for c in chroma_client.list_collections()}
    if collection_name in existing:
        collection = chroma_client.get_collection(collection_name)
        if (collection.metadata or {}).get("fingerprint") == fp and collection.count() == len(catalog):
            return collection, False
        logger.info("Catalog or knowledge base changed; rebuilding index '%s'", collection_name)
        chroma_client.delete_collection(collection_name)

    collection = chroma_client.create_collection(
        collection_name,
        embedding_function=None,  # we pass our own vectors
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"fingerprint": fp, "embedder": embedder.name},
    )
    for start in range(0, len(catalog), BATCH_SIZE):
        batch = catalog[start:start + BATCH_SIZE]
        docs = documents[start:start + BATCH_SIZE]
        collection.add(
            ids=[str(s.id) for s in batch],
            documents=docs,
            embeddings=embedder.embed(docs),
            metadatas=[song_metadata(s) for s in batch],
        )
    logger.info("Indexed %d songs into '%s' with %s", len(catalog), collection_name, embedder.name)
    return collection, True
