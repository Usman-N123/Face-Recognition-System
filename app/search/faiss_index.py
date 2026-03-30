"""
FAISS Vector Search Module — high-speed similarity search.

Manages a FAISS IndexFlatIP (inner-product) index over L2-normalized
ArcFace embeddings. Inner product on unit vectors = cosine similarity.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

import config


class FaissSearch:
    """
    FAISS-based vector search for face embeddings.

    Uses IndexFlatIP (inner product ≡ cosine similarity on L2-normed vectors).
    Maintains a mapping from FAISS internal indices to face_ids.
    """

    def __init__(
        self,
        dim: int = None,
        index_path: Path = None,
        id_map_path: Path = None,
    ):
        self._dim = dim or config.EMBEDDING_DIM
        self._index_path = str(index_path or config.FAISS_INDEX_PATH)
        self._id_map_path = str(id_map_path or config.FAISS_ID_MAP_PATH)
        self._face_ids: List[str] = []
        self._index: faiss.IndexFlatIP = None
        self._load_or_create()

    def _load_or_create(self):
        """Load existing index from disk, or create a new one."""
        index_file = Path(self._index_path)
        map_file = Path(self._id_map_path)

        if index_file.exists() and map_file.exists():
            try:
                self._index = faiss.read_index(str(index_file))
                with open(str(map_file), "r") as f:
                    self._face_ids = json.load(f)
                return
            except Exception:
                pass  # Fall through to create new

        self._index = faiss.IndexFlatIP(self._dim)
        self._face_ids = []

    def add(self, face_id: str, embedding: np.ndarray):
        """
        Add a single embedding to the index.

        Args:
            face_id: Unique face identifier.
            embedding: L2-normalized 512-d vector.
        """
        emb = embedding.astype(np.float32).reshape(1, -1)
        # Ensure L2-normalized
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        self._index.add(emb)
        self._face_ids.append(face_id)

    def add_batch(self, face_ids: List[str], embeddings: np.ndarray):
        """
        Add a batch of embeddings to the index.

        Args:
            face_ids: List of face identifiers (same order as embeddings).
            embeddings: N×512 matrix of L2-normalized vectors.
        """
        embs = embeddings.astype(np.float32)
        # Ensure L2-normalized
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embs = embs / norms
        self._index.add(embs)
        self._face_ids.extend(face_ids)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        threshold: float = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for the top-K most similar faces.

        Args:
            query_embedding: L2-normalized 512-d query vector.
            top_k: Number of nearest neighbors to retrieve.
            threshold: Minimum cosine similarity to include.

        Returns:
            List of (face_id, similarity_score) tuples, sorted by similarity desc.
        """
        if self._index.ntotal == 0:
            return []

        k = min(top_k or config.FAISS_TOP_K, self._index.ntotal)
        thresh = threshold or config.SIMILARITY_THRESHOLD

        query = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        distances, indices = self._index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._face_ids):
                continue
            similarity = float(dist)  # Inner product = cosine sim for unit vectors
            if similarity >= thresh:
                results.append((self._face_ids[idx], similarity))

        return results

    def rebuild(self, face_ids: List[str], embeddings: np.ndarray):
        """
        Rebuild the entire index from scratch.

        Args:
            face_ids: All face identifiers.
            embeddings: N×512 matrix of embeddings.
        """
        self._index = faiss.IndexFlatIP(self._dim)
        self._face_ids = []
        if len(face_ids) > 0 and embeddings.size > 0:
            self.add_batch(face_ids, embeddings)

    def save(self):
        """Persist the index and ID map to disk."""
        Path(self._index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, self._index_path)
        with open(self._id_map_path, "w") as f:
            json.dump(self._face_ids, f)

    @property
    def total(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal

    @property
    def face_ids(self) -> List[str]:
        """All face IDs in index order."""
        return self._face_ids.copy()
