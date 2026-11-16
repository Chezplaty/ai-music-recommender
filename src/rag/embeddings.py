"""Text -> vector embeddings.

We compute embeddings ourselves and hand the vectors to ChromaDB (instead of
letting Chroma embed for us). That keeps the model choice in one place and lets
tests swap in a tiny fake embedder that needs no model download.
"""

from typing import List, Protocol

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # small (80MB), fast on CPU, 384-dim vectors


class Embedder(Protocol):
    """Anything with a `name` and an `embed(texts)` method can be used for the index."""
    name: str

    def embed(self, texts: List[str]) -> List[List[float]]: ...


class SentenceTransformerEmbedder:
    """Embeds text with a sentence-transformers model. The model loads on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.name = model_name
        self._model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # slow import, so do it lazily
            try:
                # Use the cached copy without contacting Hugging Face (faster, works offline).
                self._model = SentenceTransformer(self.name, local_files_only=True)
            except OSError:
                self._model = SentenceTransformer(self.name)  # first run: download (~80MB)
        # Normalized vectors make cosine similarity well behaved (1.0 = identical direction).
        return self._model.encode(texts, normalize_embeddings=True).tolist()
