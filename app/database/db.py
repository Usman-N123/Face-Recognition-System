"""
Face Database Module — SQLite metadata storage.

Stores image records and per-face metadata including embeddings,
bounding boxes, detection scores, and cluster assignments.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config


class FaceDatabase:
    """
    SQLite database for face metadata storage.

    Schema:
        images  — one row per uploaded image (image_id, path, timestamp)
        faces   — one row per detected face (face_id, image_id, embedding, bbox, etc.)

    Thread safety: each call creates its own connection for thread-safe
    FastAPI usage. For heavy concurrent writes, consider WAL mode.
    """

    def __init__(self, db_path: Path = None):
        self._db_path = str(db_path or config.DB_PATH)
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Create a new connection with WAL mode for concurrent reads."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS images (
                    image_id    TEXT PRIMARY KEY,
                    original_path TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS faces (
                    face_id     TEXT PRIMARY KEY,
                    image_id    TEXT NOT NULL REFERENCES images(image_id),
                    embedding   BLOB NOT NULL,
                    bbox        TEXT NOT NULL,
                    det_score   REAL NOT NULL,
                    cluster_id  INTEGER DEFAULT NULL,
                    created_at  TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_faces_image_id
                    ON faces(image_id);
                CREATE INDEX IF NOT EXISTS idx_faces_cluster_id
                    ON faces(cluster_id);
            """)
            conn.commit()
        finally:
            conn.close()

    # ─── Image Operations ─────────────────────────────────────────────────

    def add_image(self, image_id: str, original_path: str) -> str:
        """
        Record a new uploaded image.

        Args:
            image_id: Unique image identifier.
            original_path: Path where the full image is stored.

        Returns:
            The image_id.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO images (image_id, original_path, uploaded_at) VALUES (?, ?, ?)",
                (image_id, original_path, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            return image_id
        finally:
            conn.close()

    def get_image(self, image_id: str) -> Optional[Dict]:
        """Retrieve an image record by ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM images WHERE image_id = ?", (image_id,)
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def image_exists(self, image_id: str) -> bool:
        """Check if an image is already in the database."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM images WHERE image_id = ?", (image_id,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    # ─── Face Operations ──────────────────────────────────────────────────

    def add_face(
        self,
        image_id: str,
        embedding: np.ndarray,
        bbox: List[int],
        det_score: float,
        face_id: str = None,
    ) -> str:
        """
        Store a detected face with its embedding and metadata.

        Args:
            image_id: ID of the parent image.
            embedding: 512-d numpy embedding vector.
            bbox: Bounding box as [x, y, w, h].
            det_score: Detection confidence score.
            face_id: Optional custom face ID (auto-generated if None).

        Returns:
            The face_id.
        """
        fid = face_id or str(uuid.uuid4())
        emb_blob = embedding.astype(np.float32).tobytes()
        bbox_json = json.dumps(bbox)
        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO faces
                   (face_id, image_id, embedding, bbox, det_score, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (fid, image_id, emb_blob, bbox_json, det_score, now),
            )
            conn.commit()
            return fid
        finally:
            conn.close()

    def get_face(self, face_id: str) -> Optional[Dict]:
        """Retrieve a face record by ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM faces WHERE face_id = ?", (face_id,)
            ).fetchone()
            if row is None:
                return None
            return self._parse_face_row(row)
        finally:
            conn.close()

    def get_faces_for_image(self, image_id: str) -> List[Dict]:
        """Get all faces detected in a specific image."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM faces WHERE image_id = ?", (image_id,)
            ).fetchall()
            return [self._parse_face_row(r) for r in rows]
        finally:
            conn.close()

    def get_all_faces(self) -> List[Dict]:
        """Retrieve all face records."""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM faces").fetchall()
            return [self._parse_face_row(r) for r in rows]
        finally:
            conn.close()

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Load all face embeddings from the database.

        Returns:
            Tuple of (face_ids, embeddings_matrix) where embeddings is N×512.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT face_id, embedding FROM faces").fetchall()
            if not rows:
                return [], np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)

            face_ids = []
            embeddings = []
            for row in rows:
                face_ids.append(row["face_id"])
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                embeddings.append(emb)

            return face_ids, np.vstack(embeddings)
        finally:
            conn.close()

    # ─── Cluster Operations ───────────────────────────────────────────────

    def update_cluster_ids(self, face_cluster_map: Dict[str, int]):
        """
        Batch update cluster_id for multiple faces.

        Args:
            face_cluster_map: {face_id: cluster_id} mapping.
        """
        conn = self._get_conn()
        try:
            for face_id, cluster_id in face_cluster_map.items():
                conn.execute(
                    "UPDATE faces SET cluster_id = ? WHERE face_id = ?",
                    (int(cluster_id), face_id),
                )
            conn.commit()
        finally:
            conn.close()

    def get_faces_by_cluster(self, cluster_id: int) -> List[Dict]:
        """Get all faces belonging to a specific cluster."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM faces WHERE cluster_id = ?", (cluster_id,)
            ).fetchall()
            return [self._parse_face_row(r) for r in rows]
        finally:
            conn.close()

    def get_image_ids_for_cluster(self, cluster_id: int) -> List[str]:
        """Get all unique image_ids that contain faces from a cluster."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT image_id FROM faces WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchall()
            return [row["image_id"] for row in rows]
        finally:
            conn.close()

    def get_images_for_cluster(self, cluster_id: int) -> List[Dict]:
        """Get full image records for all images in a cluster."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT DISTINCT i.*
                   FROM images i
                   JOIN faces f ON i.image_id = f.image_id
                   WHERE f.cluster_id = ?""",
                (cluster_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_cluster_id_for_face(self, face_id: str) -> Optional[int]:
        """Get the cluster_id assigned to a specific face."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT cluster_id FROM faces WHERE face_id = ?", (face_id,)
            ).fetchone()
            if row is None:
                return None
            return row["cluster_id"]
        finally:
            conn.close()

    def get_all_cluster_ids(self) -> List[int]:
        """Get all distinct non-null cluster IDs."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT cluster_id FROM faces WHERE cluster_id IS NOT NULL"
            ).fetchall()
            return [row["cluster_id"] for row in rows]
        finally:
            conn.close()

    def get_face_count(self) -> int:
        """Get total number of faces stored."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM faces").fetchone()
            return row["cnt"]
        finally:
            conn.close()

    def get_image_count(self) -> int:
        """Get total number of images stored."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM images").fetchone()
            return row["cnt"]
        finally:
            conn.close()

    # ─── Private Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _parse_face_row(row: sqlite3.Row) -> Dict:
        """Convert a face DB row to a dictionary with parsed fields."""
        d = dict(row)
        # Parse embedding blob back to numpy
        d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32).copy()
        # Parse bbox JSON
        d["bbox"] = json.loads(d["bbox"])
        return d
