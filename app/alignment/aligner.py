"""
Face Alignment Module — Landmark-based affine alignment.

Aligns detected faces using the 5-point facial landmarks so that
eyes are horizontal. Outputs a 112×112 aligned face crop suitable
for ArcFace embedding extraction.
"""

from typing import Optional

import cv2
import numpy as np

# Standard reference landmarks for 112×112 ArcFace input.
# These are the canonical positions InsightFace uses.
ARCFACE_REF_LANDMARKS = np.array([
    [38.2946, 51.6963],   # Left eye
    [73.5318, 51.5014],   # Right eye
    [56.0252, 71.7366],   # Nose tip
    [41.5493, 92.3655],   # Left mouth corner
    [70.7299, 92.2041],   # Right mouth corner
], dtype=np.float32)


class FaceAligner:
    """
    Aligns a face using 5-point landmarks via an affine (similarity) transform.

    The alignment maps the detected landmarks to canonical reference positions,
    producing a 112×112 face image with eyes horizontally aligned.
    """

    def __init__(self, output_size: int = 112):
        self._output_size = output_size
        self._ref_landmarks = ARCFACE_REF_LANDMARKS.copy()

    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Align a face using 5-point landmarks.

        Args:
            image: Full BGR image containing the face.
            landmarks: 5×2 array of facial landmark coordinates
                       [left_eye, right_eye, nose, left_mouth, right_mouth].

        Returns:
            Aligned 112×112 BGR face image, or None if alignment fails.
        """
        if landmarks is None or landmarks.shape != (5, 2):
            return None

        try:
            # Estimate similarity transform (rotation, scale, translation)
            transform_matrix = self._estimate_transform(landmarks)
            if transform_matrix is None:
                return None

            # Apply affine warp
            aligned = cv2.warpAffine(
                image,
                transform_matrix,
                (self._output_size, self._output_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return aligned

        except Exception:
            return None

    def _estimate_transform(self, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate a 2×3 affine transform matrix from source landmarks
        to reference landmarks using least-squares.

        Uses a similarity transform (rotation + uniform scale + translation)
        to avoid shearing.
        """
        src = landmarks.astype(np.float32)
        dst = self._ref_landmarks.astype(np.float32)

        # Solve for similarity transform: [s*cos(θ), -s*sin(θ), tx]
        #                                 [s*sin(θ),  s*cos(θ), ty]
        # Using the Umeyama algorithm (simplified)
        n = src.shape[0]

        # Center points
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        # Covariance
        cov = dst_centered.T @ src_centered / n

        # SVD
        U, S, Vt = np.linalg.svd(cov)

        # Rotation
        d = np.linalg.det(U) * np.linalg.det(Vt)
        D = np.array([[1, 0], [0, 1 if d > 0 else -1]], dtype=np.float32)
        R = U @ D @ Vt

        # Scale
        src_var = np.sum(src_centered ** 2) / n
        scale = np.sum(S * np.diag(D)) / src_var if src_var > 0 else 1.0

        # Translation
        t = dst_mean - scale * R @ src_mean

        # Build 2×3 matrix
        M = np.zeros((2, 3), dtype=np.float32)
        M[:2, :2] = scale * R
        M[:, 2] = t

        return M

    @staticmethod
    def align_simple(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> Optional[np.ndarray]:
        """
        Quick alignment using cv2.estimateAffinePartial2D.
        Fallback method if the Umeyama approach has issues.
        """
        if landmarks is None or landmarks.shape != (5, 2):
            return None

        src = landmarks.astype(np.float32)
        dst = ARCFACE_REF_LANDMARKS.astype(np.float32)

        M, _ = cv2.estimateAffinePartial2D(src, dst)
        if M is None:
            return None

        aligned = cv2.warpAffine(
            image, M, (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned
