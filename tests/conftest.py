import hashlib
import math
import re
from pathlib import Path

import pytest

from recommender import load_catalog

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FAKE_DIM = 512


@pytest.fixture(scope="session")
def catalog():
    return load_catalog(str(DATA_DIR / "catalog.csv"))


class FakeEmbedder:
    """Stands in for sentence-transformers in tests: hashes words into a normalized
    count vector, so texts sharing words get a higher cosine similarity. Offline and instant."""

    def __init__(self, name="fake-bow"):
        self.name = name

    def embed(self, texts):
        vectors = []
        for text in texts:
            v = [0.0] * FAKE_DIM
            for word in re.findall(r"[a-z&]+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % FAKE_DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def chroma_client(tmp_path):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(tmp_path), settings=Settings(anonymized_telemetry=False))


@pytest.fixture
def retriever(catalog, fake_embedder, chroma_client):
    from rag.retrieval import build_retriever
    return build_retriever(catalog, embedder=fake_embedder, chroma_client=chroma_client)
