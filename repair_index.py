"""
repair_index.py — Clean up duplicates and reindex everything from scratch.

Run this while the server is NOT running:
    python repair_index.py

This will:
1. Reset the database, FAISS index, and watcher registry
2. Reindex ALL images from images/ folder with PIL fallback for .jfif files
3. Run DBSCAN reclustering
"""

import sys
import sqlite3
import shutil
import uuid
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import config
from app.database.db import FaceDatabase
from app.clustering.clusterer import FaceClusterer
from app.detection.detector import FaceDetector
from app.embedding.embedder import FaceEmbedder
from app.utils.quality import passes_quality_check

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".jiff"}


def safe_read_image(image_path: Path):
    """
    Read an image using cv2, with PIL fallback for formats like .jfif.
    Returns a BGR numpy array or None on failure.
    """
    import cv2
    import numpy as np

    # Try cv2 first
    img = cv2.imread(str(image_path))
    if img is not None:
        return img

    # Fallback: PIL (handles .jfif, .jiff, and other edge cases reliably)
    try:
        from PIL import Image
        pil_img = Image.open(str(image_path)).convert("RGB")
        arr = np.array(pil_img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        logger.info("  [PIL fallback used] %s", image_path.name)
        return bgr
    except Exception as e:
        logger.warning("  [PIL fallback failed] %s: %s", image_path.name, e)
        return None


def step1_reset():
    """Clear DB, FAISS index, stored images, and registry."""
    logger.info("=" * 60)
    logger.info("STEP 1: Resetting all data...")
    logger.info("=" * 60)

    db_path = Path(config.DB_PATH)
    if db_path.exists():
        db_path.unlink()
        logger.info("  Deleted database: %s", db_path)

    for p in [config.FAISS_INDEX_PATH, config.FAISS_ID_MAP_PATH]:
        p = Path(str(p))
        if p.exists():
            p.unlink()
            logger.info("  Deleted FAISS file: %s", p)

    if config.IMAGES_DIR.exists():
        count = 0
        for f in config.IMAGES_DIR.iterdir():
            if f.is_file():
                f.unlink()
                count += 1
        logger.info("  Deleted %d stored images from data/images/", count)

    registry = config.DATA_DIR / "indexed_files.txt"
    if registry.exists():
        registry.unlink()
        logger.info("  Deleted registry: %s", registry)

    logger.info("  Reset complete.")


def step2_reindex():
    """Reindex all images from images/ source folder."""
    logger.info("=" * 60)
    logger.info("STEP 2: Loading models...")
    logger.info("=" * 60)

    detector = FaceDetector.get_instance()
    db = FaceDatabase()

    from app.search.faiss_index import FaissSearch
    faiss_idx = FaissSearch()

    source_dir = config.BASE_DIR / "images"
    if not source_dir.exists():
        logger.error("images/ folder not found at %s", source_dir)
        return

    image_files = sorted([
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    logger.info("=" * 60)
    logger.info("STEP 3: Indexing %d images...", len(image_files))
    logger.info("=" * 60)

    total_stored = 0
    total_skipped = 0
    indexed_names = []

    for img_path in image_files:
        image = safe_read_image(img_path)
        if image is None:
            logger.warning("  SKIP (unreadable): %s", img_path.name)
            continue

        image_id = str(uuid.uuid4())[:12]
        import cv2
        saved_path = config.IMAGES_DIR / f"{image_id}.jpg"
        cv2.imwrite(str(saved_path), image)
        db.add_image(image_id, str(saved_path))

        faces = detector.detect(image)
        stored = 0
        skipped = 0

        for face in faces:
            if not passes_quality_check(face):
                skipped += 1
                continue
            embedding = FaceEmbedder.extract_from_detected(face)
            if embedding is None:
                skipped += 1
                continue
            face_id = db.add_face(
                image_id=image_id,
                embedding=embedding,
                bbox=face.bbox_xywh,
                det_score=face.det_score,
            )
            faiss_idx.add(face_id, embedding)
            stored += 1

        total_stored += stored
        total_skipped += skipped
        indexed_names.append(img_path.name)

        status = "✅" if stored > 0 else "⚠️ "
        logger.info(
            "  %s %s — %d face(s) stored, %d skipped",
            status, img_path.name, stored, skipped
        )

    faiss_idx.save()
    logger.info("  FAISS index saved with %d vectors.", faiss_idx.total)

    # Write registry
    registry_path = config.DATA_DIR / "indexed_files.txt"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("\n".join(sorted(indexed_names)), encoding="utf-8")
    logger.info("  Registry written: %d files.", len(indexed_names))

    logger.info("=" * 60)
    logger.info("STEP 4: Reclustering...")
    logger.info("=" * 60)

    clusterer = FaceClusterer()
    cluster_map = clusterer.assign_clusters(db)
    valid_clusters = {k: v for k, v in cluster_map.items() if k >= 0}
    noise = cluster_map.get(-1, [])

    logger.info("  Clusters found: %d", len(valid_clusters))
    logger.info("  Noise faces: %d", len(noise))
    for cid, face_ids in sorted(valid_clusters.items()):
        logger.info("    Cluster %d: %d face(s)", cid, len(face_ids))

    logger.info("=" * 60)
    logger.info("DONE! Summary:")
    logger.info("  Images indexed:  %d / %d", len(indexed_names), len(image_files))
    logger.info("  Faces stored:    %d", total_stored)
    logger.info("  Faces skipped:   %d", total_skipped)
    logger.info("  Clusters found:  %d", len(valid_clusters))
    logger.info("=" * 60)


if __name__ == "__main__":
    step1_reset()
    step2_reindex()
    print("\nRepair complete! You can now start the server with: python main.py")
