"""
FastAPI Routes — REST API for face-based image retrieval.

Endpoints:
    POST /upload     — Upload image, detect/align/embed/store faces
    POST /query      — Query with face image, retrieve matching full images
    GET  /results/{cluster_id} — Get all images for a person cluster
    POST /recluster  — Re-run DBSCAN clustering on all stored embeddings
    GET  /health     — Health check
"""

import shutil
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import config
from app.clustering.clusterer import FaceClusterer
from app.database.db import FaceDatabase
from app.detection.detector import FaceDetector
from app.embedding.embedder import FaceEmbedder
from app.search.faiss_index import FaissSearch
from app.utils.quality import passes_quality_check

router = APIRouter()

# ─── Lazy-initialized singletons ──────────────────────────────────────────────
_db: FaceDatabase = None
_faiss: FaissSearch = None
_detector: FaceDetector = None
_embedder: FaceEmbedder = None
_clusterer: FaceClusterer = None


def get_db() -> FaceDatabase:
    global _db
    if _db is None:
        _db = FaceDatabase()
    return _db


def get_faiss() -> FaissSearch:
    global _faiss
    if _faiss is None:
        _faiss = FaissSearch()
    return _faiss


def get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector.get_instance()
    return _detector


def get_embedder() -> FaceEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = FaceEmbedder()
    return _embedder


def get_clusterer() -> FaceClusterer:
    global _clusterer
    if _clusterer is None:
        _clusterer = FaceClusterer()
    return _clusterer


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _read_image(file: UploadFile) -> np.ndarray:
    """Read an uploaded file into a BGR numpy array."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        # Fallback: try PIL (handles .jfif and unusual encodings)
        try:
            from PIL import Image
            import io
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            arr = np.array(pil_img)
            image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return image


def _safe_read_image_file(image_path) -> "np.ndarray | None":
    """
    Read an image from disk using cv2, with PIL as a fallback.
    Handles .jfif/.jiff and other formats that cv2 may not decode on all platforms.
    """
    from pathlib import Path as _Path
    img = cv2.imread(str(image_path))
    if img is not None:
        return img
    try:
        from PIL import Image
        pil_img = Image.open(str(image_path)).convert("RGB")
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _save_image(image: np.ndarray, image_id: str) -> str:
    """Save full image to data/images/ and return the path."""
    save_path = config.IMAGES_DIR / f"{image_id}.jpg"
    cv2.imwrite(str(save_path), image)
    return str(save_path)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image → detect all faces → align → embed → store.

    Returns:
        JSON with image_id, number of faces detected, face details.
    """
    image = await _read_image(file)
    db = get_db()
    faiss_idx = get_faiss()
    detector = get_detector()

    # Generate image ID and save full image
    image_id = str(uuid.uuid4())[:12]
    saved_path = _save_image(image, image_id)
    db.add_image(image_id, saved_path)

    # Detect all faces
    faces = detector.detect(image)

    stored_faces = []
    skipped = 0

    for face in faces:
        # Quality gate
        if not passes_quality_check(face):
            skipped += 1
            continue

        # Extract embedding (already computed by InsightFace)
        embedding = FaceEmbedder.extract_from_detected(face)
        if embedding is None:
            skipped += 1
            continue

        # Store in database
        face_id = db.add_face(
            image_id=image_id,
            embedding=embedding,
            bbox=face.bbox_xywh,
            det_score=face.det_score,
        )

        # Add to FAISS index
        faiss_idx.add(face_id, embedding)

        stored_faces.append({
            "face_id": face_id,
            "bbox": face.bbox_xywh,
            "det_score": round(face.det_score, 4),
        })

    # Save FAISS index to disk
    faiss_idx.save()

    return {
        "image_id": image_id,
        "image_path": saved_path,
        "faces_detected": len(faces),
        "faces_stored": len(stored_faces),
        "faces_skipped": skipped,
        "faces": stored_faces,
    }


@router.post("/query")
async def query_face(
    file: UploadFile = File(...),
    top_k: int = None,
    threshold: float = None,
):
    """
    Upload a face image → detect → embed → FAISS search → cluster voting
    → return all full images containing that person.

    The query image should contain one primary face (the person to search for).

    Returns:
        JSON with query details, matched cluster, and all matching image paths.
    """
    image = await _read_image(file)
    db = get_db()
    faiss_idx = get_faiss()
    detector = get_detector()
    clusterer = get_clusterer()

    # Detect the largest face in query image
    query_face = detector.detect_largest(image)
    if query_face is None:
        raise HTTPException(status_code=400, detail="No face detected in query image")

    # Extract embedding
    query_embedding = FaceEmbedder.extract_from_detected(query_face)
    if query_embedding is None:
        raise HTTPException(status_code=400, detail="Failed to extract embedding from query face")

    # FAISS search
    k = top_k or config.FAISS_TOP_K
    thresh = threshold or config.SIMILARITY_THRESHOLD
    matches = faiss_idx.search(query_embedding, top_k=k, threshold=thresh)

    if not matches:
        return {
            "query_bbox": query_face.bbox_xywh,
            "matches_found": 0,
            "cluster_id": None,
            "images": [],
            "message": "No matching faces found",
        }

    # Extract matched face_ids and their similarities
    matched_face_ids = [fid for fid, _ in matches]

    # --- Cluster-based retrieval (primary) ---
    best_cluster = clusterer.majority_vote(matched_face_ids, db)
    seen_images = set()
    result_images = []

    if best_cluster is not None:
        cluster_images = db.get_images_for_cluster(best_cluster)
        for img in cluster_images:
            if img["image_id"] not in seen_images:
                seen_images.add(img["image_id"])
                result_images.append({
                    "image_id": img["image_id"],
                    "image_path": img["original_path"],
                })

    # --- Direct FAISS fallback (union) ---
    # Include any extra images the FAISS search found that weren't in the cluster,
    # e.g. if a face was marked as noise (-1) by DBSCAN or is in a second cluster.
    for face_id in matched_face_ids:
        face_record = db.get_face(face_id)
        if face_record and face_record["image_id"] not in seen_images:
            seen_images.add(face_record["image_id"])
            img_record = db.get_image(face_record["image_id"])
            if img_record:
                result_images.append({
                    "image_id": img_record["image_id"],
                    "image_path": img_record["original_path"],
                })

    return {
        "query_bbox": query_face.bbox_xywh,
        "matches_found": len(matches),
        "top_matches": [
            {"face_id": fid, "similarity": round(sim, 4)}
            for fid, sim in matches[:5]
        ],
        "cluster_id": best_cluster,
        "total_images": len(result_images),
        "images": result_images,
    }


@router.get("/results/{cluster_id}")
async def get_cluster_results(cluster_id: int):
    """
    Get all full images containing faces from a specific cluster (person).

    Returns:
        JSON with cluster_id, face count, and all image paths.
    """
    db = get_db()

    faces = db.get_faces_by_cluster(cluster_id)
    if not faces:
        raise HTTPException(status_code=404, detail=f"No faces found for cluster {cluster_id}")

    images = db.get_images_for_cluster(cluster_id)

    return {
        "cluster_id": cluster_id,
        "face_count": len(faces),
        "image_count": len(images),
        "images": [
            {
                "image_id": img["image_id"],
                "image_path": img["original_path"],
            }
            for img in images
        ],
    }


@router.post("/recluster")
async def recluster():
    """
    Re-run DBSCAN clustering on all stored face embeddings.

    This should be called after uploading a batch of images to
    group faces by identity.

    Returns:
        JSON with number of clusters and cluster summary.
    """
    db = get_db()
    clusterer = get_clusterer()

    cluster_faces = clusterer.assign_clusters(db)

    # Filter out noise cluster (-1)
    valid_clusters = {k: v for k, v in cluster_faces.items() if k >= 0}
    noise_count = len(cluster_faces.get(-1, []))

    summary = []
    for cid, face_ids in sorted(valid_clusters.items()):
        image_ids = db.get_image_ids_for_cluster(cid)
        summary.append({
            "cluster_id": cid,
            "face_count": len(face_ids),
            "image_count": len(image_ids),
            "image_ids": image_ids,
        })

    return {
        "total_faces": db.get_face_count(),
        "total_clusters": len(valid_clusters),
        "noise_faces": noise_count,
        "clusters": summary,
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    db = get_db()
    faiss_idx = get_faiss()
    return {
        "status": "healthy",
        "images_stored": db.get_image_count(),
        "faces_stored": db.get_face_count(),
        "faiss_vectors": faiss_idx.total,
    }


@router.get("/image/{image_id}")
async def serve_image(image_id: str):
    """Serve a stored image by its image_id."""
    db = get_db()
    record = db.get_image(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Image not found")

    path = Path(record["original_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FileResponse(str(path), media_type="image/jpeg")


@router.post("/reset")
async def reset_database():
    """
    Clear all stored images, faces, and the FAISS index.
    Use this before re-indexing with new thresholds.
    """
    global _db, _faiss

    # Drop and recreate database
    db_path = Path(config.DB_PATH)
    if db_path.exists():
        db_path.unlink()

    # Remove FAISS index files
    for p in [config.FAISS_INDEX_PATH, config.FAISS_ID_MAP_PATH]:
        if Path(str(p)).exists():
            Path(str(p)).unlink()

    # Clear stored images
    if config.IMAGES_DIR.exists():
        for f in config.IMAGES_DIR.iterdir():
            if f.is_file():
                f.unlink()

    # Clear watcher registry
    registry = config.DATA_DIR / "indexed_files.txt"
    if registry.exists():
        registry.unlink()

    # Re-initialize singletons
    _db = FaceDatabase()
    _faiss = FaissSearch()

    return {"message": "Database and FAISS index cleared.", "status": "reset"}


@router.post("/reindex")
async def reindex_images():
    """
    Re-upload and re-process all original images from the source images/ folder.
    Run this after /reset to rebuild the index with current thresholds.
    """
    db = get_db()
    faiss_idx = get_faiss()
    detector = get_detector()

    source_dir = config.BASE_DIR / "images"
    if not source_dir.exists():
        raise HTTPException(status_code=404, detail="source images/ folder not found")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".jiff"}
    image_files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions]

    if not image_files:
        raise HTTPException(status_code=404, detail="No images found in source images/ folder")

    import importlib
    import config as cfg
    importlib.reload(cfg)  # pick up any config changes

    total_stored = 0
    total_skipped = 0
    results = []

    for img_path in sorted(image_files):
        image = _safe_read_image_file(img_path)
        if image is None:
            continue

        image_id = str(uuid.uuid4())[:12]
        saved_path = str(config.IMAGES_DIR / f"{image_id}.jpg")
        cv2.imwrite(saved_path, image)
        db.add_image(image_id, saved_path)

        faces = detector.detect(image)
        stored = 0
        skipped = 0

        for face in faces:
            from app.utils.quality import passes_quality_check
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
        results.append({
            "file": img_path.name,
            "image_id": image_id,
            "faces_stored": stored,
            "faces_skipped": skipped,
        })

    faiss_idx.save()

    # Auto-cluster after indexing
    clusterer = get_clusterer()
    cluster_map = clusterer.assign_clusters(db)
    valid_clusters = {k: v for k, v in cluster_map.items() if k >= 0}

    return {
        "images_processed": len(results),
        "total_faces_stored": total_stored,
        "total_faces_skipped": total_skipped,
        "clusters_found": len(valid_clusters),
        "details": results,
    }


@router.get("/watcher-status")
async def watcher_status():
    """
    Return the count of source images registered in the auto-index registry.
    Used by the frontend to display how many photos are indexed without exposing
    manual reset/re-index controls.
    """
    registry_path = config.DATA_DIR / "indexed_files.txt"
    if not registry_path.exists():
        return {"indexed_count": 0, "indexed_files": []}
    files = [f for f in registry_path.read_text(encoding="utf-8").splitlines() if f]
    return {"indexed_count": len(files), "indexed_files": files}
