"""
utils/rag_engine.py
-------------------
Implements the RAG (Retrieval-Augmented Generation) pipeline:
  1. Segment document text into logical clauses / chunks.
  2. Embed each chunk with SentenceTransformers.
  3. Build a FAISS index for fast similarity search.
  4. Retrieve the top-k most relevant chunks for a user query.
"""

import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Model (loaded once, reused across calls)
# ---------------------------------------------------------------------------

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


# ---------------------------------------------------------------------------
# Clause segmentation
# ---------------------------------------------------------------------------

def segment_clauses(text: str, max_chunk_len: int = 400) -> List[str]:
    """
    Split document text into logical chunks / clauses.

    Strategy:
      - Split on double newlines (paragraph breaks).
      - Further split long paragraphs on sentence boundaries.
      - Filter out very short fragments (< 20 chars).

    Args:
        text:          Full extracted document text.
        max_chunk_len: Maximum character length for a single chunk.

    Returns:
        List of clause strings.
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on paragraph breaks
    paragraphs = re.split(r"\n{2,}", text)

    clauses: List[str] = []
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_chunk_len:
            if len(para) >= 20:
                clauses.append(para)
        else:
            # Break long paragraphs into sentences
            sentences = sentence_pattern.split(para)
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) + 1 <= max_chunk_len:
                    current_chunk = (current_chunk + " " + sent).strip()
                else:
                    if current_chunk and len(current_chunk) >= 20:
                        clauses.append(current_chunk)
                    current_chunk = sent.strip()
            if current_chunk and len(current_chunk) >= 20:
                clauses.append(current_chunk)

    return clauses


# ---------------------------------------------------------------------------
# FAISS index management
# ---------------------------------------------------------------------------

class FAISSIndex:
    """Wraps a FAISS flat L2 index with the corresponding clause list."""

    def __init__(self):
        self.index: faiss.IndexFlatL2 | None = None
        self.clauses: List[str] = []

    def build(self, clauses: List[str]) -> None:
        """
        Embed all clauses and build the FAISS index.

        Args:
            clauses: List of clause strings to index.
        """
        self.clauses = clauses
        model = _get_embed_model()
        embeddings = model.encode(clauses, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve the top-k most relevant clauses for a query.

        Args:
            query: User's natural-language question.
            top_k: Number of results to return.

        Returns:
            List of (clause_text, distance) tuples, sorted by relevance.
        """
        if self.index is None or not self.clauses:
            return []

        model = _get_embed_model()
        q_emb = model.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")

        distances, indices = self.index.search(q_emb, min(top_k, len(self.clauses)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.clauses):
                results.append((self.clauses[idx], float(dist)))

        return results


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def build_index_from_text(text: str) -> FAISSIndex:
    """
    Convenience function: segment → embed → index.

    Args:
        text: Full document text.

    Returns:
        Populated FAISSIndex instance.
    """
    clauses = segment_clauses(text)
    idx = FAISSIndex()
    idx.build(clauses)
    return idx
