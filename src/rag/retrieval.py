"""Semantic search over the song index, returning real catalog Song objects."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from recommender import Song
from rag.embeddings import Embedder, SentenceTransformerEmbedder
from rag.ingestion import PROJECT_ROOT, ensure_index

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = PROJECT_ROOT / ".chroma"


@dataclass
class RetrievedSong:
    song: Song
    similarity: float  # cosine similarity between query and song document, higher = closer
    document: str      # the text that matched; this is the evidence for the retrieval


def is_excluded(song: Song, exclude_artists: Iterable[str]) -> bool:
    """Case-insensitive substring match, so 'weeknd' also excludes 'The Weeknd, Daft Punk'."""
    artist = song.artist.lower()
    return any(x.lower() in artist for x in exclude_artists if x)


class MusicRetriever:
    def __init__(self, collection, catalog: List[Song], embedder: Embedder):
        self.collection = collection
        self.embedder = embedder
        self.by_id = {s.id: s for s in catalog}

    def search(self, query: str, n_results: int = 30, exclude_artists: Iterable[str] = ()) -> List[RetrievedSong]:
        """Top `n_results` catalog songs whose documents are semantically closest to `query`."""
        if not query or not query.strip():
            return []
        exclude_artists = list(exclude_artists)

        where = None
        if exclude_artists:
            # Filter inside Chroma so exclusions don't eat into n_results.
            allowed = [s.id for s in self.by_id.values() if not is_excluded(s, exclude_artists)]
            if not allowed:
                return []
            where = {"song_id": {"$in": allowed}}

        n = min(n_results, self.collection.count())
        if n <= 0:
            return []
        res = self.collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=n,
            where=where,
            include=["documents", "distances"],
        )

        results = []
        for id_, doc, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
            song = self.by_id.get(int(id_))
            if song is None:
                # Guardrail: never surface something that isn't in the current catalog
                # (e.g. a stale index). Every result maps to a real catalog row.
                logger.warning("Retrieved id %s is not in the catalog; skipping", id_)
                continue
            results.append(RetrievedSong(song=song, similarity=1.0 - dist, document=doc))
        return results


def build_retriever(catalog: List[Song], persist_dir: Optional[Path] = None,
                    embedder: Optional[Embedder] = None, chroma_client=None) -> MusicRetriever:
    """Open (or build on first run) the persistent ChromaDB index for `catalog`."""
    embedder = embedder or SentenceTransformerEmbedder()
    if chroma_client is None:
        import chromadb
        from chromadb.config import Settings
        chroma_client = chromadb.PersistentClient(
            path=str(persist_dir or DEFAULT_PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    collection, rebuilt = ensure_index(chroma_client, catalog, embedder)
    if rebuilt:
        logger.info("Built a new music index (%d songs)", collection.count())
    return MusicRetriever(collection, catalog, embedder)
