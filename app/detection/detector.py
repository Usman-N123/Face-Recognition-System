"""
Face Detection Module — RetinaFace via InsightFace.

Detects all faces in an image, returning bounding boxes, landmarks,
detection scores, and cropped face regions.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import config


@dataclass
class DetectedFace:
    """Container for a single detected face."""
    bbox: np.ndarray            # [x1, y1, x2, y2]
    landmarks: np.ndarray       # 5-point landmarks (2D, shape 5×2)
    det_score: float            # Detection confidence [0, 1]
    face_crop: np.ndarray       # Raw face crop from the original image
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def width(self) -> int:
        """Width of the bounding box in pixels."""
        return int(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        """Height of the bounding box in pixels."""
        return int(self.bbox[3] - self.bbox[1])

    @property
    def bbox_xywh(self) -> List[int]:
        """Bounding box as [x, y, w, h]."""
        x1, y1, x2, y2 = self.bbox.astype(int)
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


class FaceDetector:
    """
    RetinaFace-based face detector using InsightFace's FaceAnalysis.

    On first instantiation, downloads the buffalo_l model pack (~300 MB)
    to ~/.insightface/models/ if not already cached.
    """

    _instance: Optional["FaceDetector"] = None

    def __init__(self, model_name: str = None, det_size: tuple = None):
        self._model_name = model_name or config.MODEL_NAME
        self._det_size = det_size or config.DET_SIZE
        self._app = FaceAnalysis(
            name=self._model_name,
            providers=config.PROVIDERS,
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=0, det_size=self._det_size)

    @classmethod
    def get_instance(cls) -> "FaceDetector":
        """Singleton accessor to avoid reloading models."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in a BGR image.

        Args:
            image: BGR numpy array (as returned by cv2.imread).

        Returns:
            List of DetectedFace objects, filtered by minimum detection score.
        """
        if image is None or image.size == 0:
            return []

        faces = self._app.get(image)
        results: List[DetectedFace] = []

        for face in faces:
            # Filter by detection score
            if face.det_score < config.MIN_DET_SCORE:
                continue

            bbox = face.bbox.astype(np.float32)
            landmarks = face.landmark_2d_106 if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None else face.kps

            # Crop face from original image (with boundary clamping)
            h, w = image.shape[:2]
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))
            face_crop = image[y1:y2, x1:x2].copy()

            detected = DetectedFace(
                bbox=bbox,
                landmarks=face.kps,  # 5-point landmarks for alignment
                det_score=float(face.det_score),
                face_crop=face_crop,
                embedding=face.normed_embedding if hasattr(face, "normed_embedding") and face.normed_embedding is not None else face.embedding,
            )
            results.append(detected)

        return results

    def detect_largest(self, image: np.ndarray) -> Optional[DetectedFace]:
        """
        Detect and return only the largest face in the image.
        Useful for query images where we expect a single face.
        """
        faces = self.detect(image)
        if not faces:
            return None
        return max(faces, key=lambda f: f.width * f.height)
