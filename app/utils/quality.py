"""
Quality Control Utilities — face quality validation.

Filters out faces that are too small, blurry, or detected with low confidence
to ensure only high-quality faces enter the embedding pipeline.
"""

import cv2
import numpy as np

import config
from app.detection.detector import DetectedFace


def is_face_large_enough(face: DetectedFace, min_width: int = None) -> bool:
    """
    Check if the detected face meets minimum size requirements.

    Args:
        face: DetectedFace object.
        min_width: Minimum acceptable width in pixels.

    Returns:
        True if the face is large enough.
    """
    min_w = min_width or config.MIN_FACE_WIDTH
    return face.width >= min_w


def is_face_sharp(face_crop: np.ndarray, threshold: float = None) -> bool:
    """
    Check if a face crop is not blurry using Laplacian variance.

    A higher Laplacian variance indicates a sharper image.
    Typical thresholds: 50–150 depending on resolution.

    Args:
        face_crop: BGR face crop.
        threshold: Minimum Laplacian variance (default from config).

    Returns:
        True if the face is sharp enough.
    """
    thresh = threshold or config.BLUR_THRESHOLD
    if face_crop is None or face_crop.size == 0:
        return False

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= thresh


def has_sufficient_confidence(face: DetectedFace, min_score: float = None) -> bool:
    """
    Check if the detection confidence is above threshold.

    Args:
        face: DetectedFace object.
        min_score: Minimum detection score (default from config).

    Returns:
        True if confidence is sufficient.
    """
    score = min_score or config.MIN_DET_SCORE
    return face.det_score >= score


def passes_quality_check(
    face: DetectedFace,
    check_blur: bool = True,
    check_size: bool = True,
    check_confidence: bool = True,
) -> bool:
    """
    Run all quality checks on a detected face.

    Args:
        face: DetectedFace object.
        check_blur: Whether to check for blurriness.
        check_size: Whether to check minimum face size.
        check_confidence: Whether to check detection confidence.

    Returns:
        True if the face passes all enabled checks.
    """
    if check_confidence and not has_sufficient_confidence(face):
        return False

    if check_size and not is_face_large_enough(face):
        return False

    if check_blur and not is_face_sharp(face.face_crop):
        return False

    return True
