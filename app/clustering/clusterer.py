"""
Face Clustering Module — DBSCAN-based identity grouping.

Clusters face embeddings so that each cluster represents one person.
Uses cosine distance (1 − cosine_similarity) with DBSCAN for
density-based clustering without requiring a predefined number of people.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

import config
from app.database.db import FaceDatabase


class FaceClusterer:
    """
    DBSCAN-based face clustering.

    Groups face embeddings into clusters where each cluster represents
    a unique identity. Handles noise points (outliers) and multi-person
    group images automatically.
    """

    def __init__(
        self,
        eps: float = None,
        min_samples: int = None,
    ):
        self._eps = eps or config.DBSCAN_EPS
        self._min_samples = min_samples or config.DBSCAN_MIN_SAMPLES

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings using DBSCAN with cosine distance.

        Args:
            embeddings: N×512 matrix of L2-normalized face embeddings.

        Returns:
            Array of cluster labels (length N). Label -1 = noise/outlier.
        """
        if embeddings.shape[0] == 0:
            return np.array([], dtype=int)

        # L2-normalize to ensure cosine distance works correctly
        normed = normalize(embeddings, norm="l2")

        # DBSCAN with precomputed cosine distance matrix
        # cosine_distance = 1 - cosine_similarity
        # For L2-normalized vectors: cosine_sim = dot product
        similarity_matrix = normed @ normed.T
        distance_matrix = 1.0 - similarity_matrix

        # Clamp to [0, 2] to avoid numerical issues
        distance_matrix = np.clip(distance_matrix, 0, 2)

        clusterer = DBSCAN(
            eps=self._eps,
            min_samples=self._min_samples,
            metric="precomputed",
        )
        labels = clusterer.fit_predict(distance_matrix)
        return labels

    def assign_clusters(self, db: FaceDatabase) -> Dict[int, List[str]]:
        """
        Load all embeddings from DB, cluster them, and write cluster_ids back.

        Args:
            db: FaceDatabase instance.

        Returns:
            Dict mapping cluster_id → [face_id1, face_id2, ...].
            Cluster -1 contains noise points.
        """
        face_ids, embeddings = db.get_all_embeddings()

        if len(face_ids) == 0:
            return {}

        labels = self.cluster(embeddings)

        # Build face_id → cluster_id mapping
        face_cluster_map = {
            face_id: int(label)
            for face_id, label in zip(face_ids, labels)
        }
        db.update_cluster_ids(face_cluster_map)

        # Build cluster_id → [face_ids] mapping
        cluster_faces: Dict[int, List[str]] = {}
        for face_id, label in zip(face_ids, labels):
            label_int = int(label)
            if label_int not in cluster_faces:
                cluster_faces[label_int] = []
            cluster_faces[label_int].append(face_id)

        return cluster_faces

    @staticmethod
    def majority_vote(
        matched_face_ids: List[str],
        db: FaceDatabase,
    ) -> Optional[int]:
        """
        Determine the best cluster_id via majority voting.

        Given a list of matched face_ids from FAISS search, count which
        cluster_id appears most frequently and return it.

        Args:
            matched_face_ids: List of face_ids from FAISS top-K results.
            db: FaceDatabase to look up cluster assignments.

        Returns:
            The cluster_id with the most votes, or None if no valid clusters.
        """
        cluster_ids = []
        for face_id in matched_face_ids:
            cid = db.get_cluster_id_for_face(face_id)
            if cid is not None and cid >= 0:  # Exclude noise (-1)
                cluster_ids.append(cid)

        if not cluster_ids:
            return None

        # Majority vote
        counter = Counter(cluster_ids)
        best_cluster, _ = counter.most_common(1)[0]
        return best_cluster

    @staticmethod
    def get_cluster_summary(db: FaceDatabase) -> List[Dict]:
        """
        Get a summary of all clusters with their face and image counts.

        Returns:
            List of {cluster_id, face_count, image_count, image_ids}.
        """
        all_clusters = db.get_all_cluster_ids()
        summaries = []
        for cid in sorted(all_clusters):
            faces = db.get_faces_by_cluster(cid)
            image_ids = list(set(f["image_id"] for f in faces))
            summaries.append({
                "cluster_id": cid,
                "face_count": len(faces),
                "image_count": len(image_ids),
                "image_ids": image_ids,
            })
        return summaries
