"""
Automatic Image Watcher — monitors the images/ source folder for new photos
and automatically ingests them into the face index without any user action.

When a photographer drops a new image into the watched folder:
  1. Image is read and saved into data/images/
  2. All faces are detected, quality-checked, and embedded
  3. Embeddings are stored in SQLite + FAISS index
  4. DBSCAN reclustering runs to update identity groups

The watcher runs as a daemon thread alongside the FastAPI server so it
never blocks request handling.
"""

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Set

import cv2
from watchdog.events import FileCreatedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

import config
from app.clustering.clusterer import FaceClusterer
from app.database.db import FaceDatabase
from app.detection.detector import FaceDetector
from app.embedding.embedder import FaceEmbedder
from app.search.faiss_index import FaissSearch
from app.utils.quality import passes_quality_check

logger = logging.getLogger(__name__)

# Extensions we will process
IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".jiff"}

# How long to wait (seconds) before processing a file to let the OS finish writing it
SETTLE_DELAY = 1.5

# Minimum file size in bytes (ignore zero-byte / incomplete writes)
MIN_FILE_SIZE = 1024


def _safe_read_image(image_path: Path):
    """
    Read an image using cv2 with PIL/Pillow as a fallback.

    cv2.imread can silently return None for .jfif / .jiff files on some
    platforms even though the file is valid JPEG data.  Pillow handles
    these reliably, so we fall back to it whenever cv2 fails.

    Returns a BGR numpy array, or None if both methods fail.
    """
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    if img is not None:
        return img

    # Fallback: Pillow (pip install Pillow)
    try:
        from PIL import Image
        pil_img = Image.open(str(image_path)).convert("RGB")
        arr = np.array(pil_img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        logger.debug("[watcher] PIL fallback used for %s", image_path.name)
        return bgr
    except Exception as e:
        logger.warning("[watcher] PIL fallback failed for %s: %s", image_path.name, e)
        return None


def _process_image(image_path: Path) -> dict:
    """
    Process a single new image: detect faces → embed → store in DB + FAISS.

    Returns a dict with processing statistics.
    """
    db = FaceDatabase()
    faiss_idx = FaissSearch()
    detector = FaceDetector.get_instance()

    image = _safe_read_image(image_path)
    if image is None:
        logger.warning("[watcher] Could not read image: %s", image_path)
        return {"file": image_path.name, "error": "unreadable"}

    # Save a copy into data/images/ (same as /upload endpoint does)
    image_id = str(uuid.uuid4())[:12]
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

    faiss_idx.save()

    # Update the registry so a server restart doesn't re-process this file
    registry_path = config.DATA_DIR / "indexed_files.txt"
    try:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        existing = set(registry_path.read_text(encoding="utf-8").splitlines()) if registry_path.exists() else set()
        existing.add(image_path.name)
        registry_path.write_text("\n".join(sorted(existing)), encoding="utf-8")
    except Exception:
        pass  # Non-critical — startup sync will catch it on next restart

    # Re-cluster so new face joins the right identity group
    if stored > 0:
        clusterer = FaceClusterer()
        clusterer.assign_clusters(db)
        logger.info(
            "[watcher] ✅ %s — %d face(s) stored, %d skipped. Reclustered.",
            image_path.name, stored, skipped,
        )
    else:
        logger.info(
            "[watcher] ⚠️  %s — No usable faces found (%d skipped).",
            image_path.name, skipped,
        )

    return {"file": image_path.name, "image_id": image_id,
            "faces_stored": stored, "faces_skipped": skipped}


class _NewImageHandler(FileSystemEventHandler):
    """
    Watchdog event handler — triggered on file creation or move into the folder.
    Uses a small debounce set so we don't process the same file twice.
    """

    def __init__(self):
        super().__init__()
        self._seen: Set[str] = set()
        self._lock = threading.Lock()

    def _should_handle(self, path: str) -> bool:
        ext = Path(path).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            return False
        with self._lock:
            if path in self._seen:
                return False
            self._seen.add(path)
        return True

    def _handle(self, path: str):
        if not self._should_handle(path):
            return
        # Run in a thread so the event loop is not blocked
        threading.Thread(target=self._delayed_process, args=(path,), daemon=True).start()

    def _delayed_process(self, path: str):
        """Wait for the file to finish writing, then process it."""
        time.sleep(SETTLE_DELAY)
        p = Path(path)
        if not p.exists() or p.stat().st_size < MIN_FILE_SIZE:
            logger.debug("[watcher] Skipping incomplete/missing file: %s", path)
            return
        try:
            _process_image(p)
        except Exception as exc:
            logger.error("[watcher] Error processing %s: %s", path, exc, exc_info=True)

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event):
        # Handles files moved/copied into the watched folder
        if not event.is_directory:
            self._handle(event.dest_path)


class ImageWatcher:
    """
    Manages the watchdog Observer lifecycle.

    Usage:
        watcher = ImageWatcher(watch_dir=config.BASE_DIR / "images")
        watcher.start()
        ...
        watcher.stop()
    """

    def __init__(self, watch_dir: Path = None):
        self._watch_dir = watch_dir or (config.BASE_DIR / "images")
        self._observer: Observer = None

    def start(self):
        """Start watching the folder in a daemon thread."""
        self._watch_dir.mkdir(parents=True, exist_ok=True)
        handler = _NewImageHandler()
        self._observer = Observer()
        self._observer.schedule(handler, str(self._watch_dir), recursive=False)
        self._observer.daemon = True
        self._observer.start()
        logger.info("[watcher] 👁 Watching for new images in: %s", self._watch_dir)

    def stop(self):
        """Gracefully stop the observer."""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join(timeout=5)
            logger.info("[watcher] Stopped.")

    def is_alive(self) -> bool:
        return self._observer is not None and self._observer.is_alive()
