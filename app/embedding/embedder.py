"""
Face Embedding Module — ArcFace via InsightFace.

Extracts L2-normalized 512-dimensional embedding vectors from aligned
face images using the ArcFace model bundled in InsightFace's buffalo_l pack.

Note: InsightFace's FaceAnalysis.get() already produces embeddings during
detection. This module provides a clean interface and handles cases where
we need to re-embed an aligned face independently.
"""

from typing import List, Optional

import numpy as np

import config
from app.detection.detector import FaceDetector, DetectedFace


class FaceEmbedder:
    """
    ArcFace embedding extractor.

    For efficiency, embeddings are extracted during detection (InsightFace
    produces them in a single pass). This class wraps that functionality
    and provides utilities for normalization and batch processing.
    """

    def __init__(self):
        self._detector = FaceDetector.get_instance()

    @staticmethod
    def normalize(embedding: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            return embedding
        return embedding / norm

    @staticmethod
    def extract_from_detected(face: DetectedFace) -> Optional[np.ndarray]:
        """
        Extract the L2-normalized embedding from a DetectedFace.

        InsightFace already computes embeddings during detection.
        This method ensures normalization.
        """
        if face.embedding is None:
            return None
        return FaceEmbedder.normalize(face.embedding)

    def embed_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect all faces in an image and return their embeddings.

        Args:
            image: BGR numpy array.

        Returns:
            List of L2-normalized 512-d embeddings.
        """
        faces = self._detector.detect(image)
        embeddings = []
        for face in faces:
            emb = self.extract_from_detected(face)
            if emb is not None:
                embeddings.append(emb)
        return embeddings

    def embed_query(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the largest face in a query image and return its embedding.

        Args:
            image: BGR numpy array (expected to contain one primary face).

        Returns:
            L2-normalized 512-d embedding, or None if no face found.
        """
        face = self._detector.detect_largest(image)
        if face is None:
            return None
        return self.extract_from_detected(face)

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two L2-normalized embeddings.
        For normalized vectors, this equals their dot product.
        """
        return float(np.dot(emb1, emb2))

    @staticmethod
    def batch_normalize(embeddings: np.ndarray) -> np.ndarray:
        """
        L2-normalize a batch of embeddings (N×D matrix).
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return embeddings / norms
