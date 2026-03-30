"""
Face-Based Image Retrieval System — FastAPI Entry Point.

Starts the API server with all routes mounted.
Serves stored images as static files at /data/images/.

Usage:
    python main.py
    # Server starts at http://localhost:8000
    # Swagger docs at http://localhost:8000/docs
"""

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import config
from app.api.routes import router
from app.watcher.image_watcher import ImageWatcher, _process_image, IMAGE_EXTENSIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global watcher instance
_watcher: ImageWatcher = None

FRONTEND_PATH = config.BASE_DIR / "frontend.html"


def _startup_sync():
    """
    On server start, scan the images/ source folder and process any files
    that are not yet registered in the database.  This means simply
    restarting the server after dropping new photos into the folder is
    sufficient — no manual reset/re-index needed.
    """
    from app.database.db import FaceDatabase
    from app.clustering.clusterer import FaceClusterer

    source_dir = config.BASE_DIR / "images"
    if not source_dir.exists():
        logger.info("[startup] images/ folder not found, nothing to sync.")
        return

    image_files = [
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_files:
        logger.info("[startup] images/ folder is empty, nothing to sync.")
        return

    db = FaceDatabase()

    # Build the set of original source filenames already in the DB to avoid
    # double-indexing.  We store the source filename in the image_id's
    # saved_path, but a simpler approach is to track a lightweight registry
    # file alongside the DB.
    registry_path = config.DATA_DIR / "indexed_files.txt"
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    indexed: set = set()
    if registry_path.exists():
        indexed = set(registry_path.read_text(encoding="utf-8").splitlines())

    new_files = [f for f in image_files if f.name not in indexed]
    if not new_files:
        logger.info("[startup] All %d image(s) already indexed. No sync needed.", len(image_files))
        return

    logger.info("[startup] Syncing %d new image(s) into the index...", len(new_files))
    newly_indexed = []
    for img_path in sorted(new_files):
        try:
            _process_image(img_path)
            newly_indexed.append(img_path.name)
        except Exception as exc:
            logger.error("[startup] Failed to process %s: %s", img_path.name, exc)

    # Persist updated registry
    all_indexed = sorted(indexed | set(newly_indexed))
    registry_path.write_text("\n".join(all_indexed), encoding="utf-8")

    if newly_indexed:
        # Final recluster over everything
        FaceClusterer().assign_clusters(db)
        logger.info("[startup] Sync complete. %d new image(s) indexed.", len(newly_indexed))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Face-Based Image Retrieval System",
        description=(
            "Upload images with faces, then query with a face photo to "
            "retrieve all full images containing that person — including group photos."
        ),
        version="1.0.0",
    )

    # CORS (allow all origins for development)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount stored images as static files
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/static/images",
        StaticFiles(directory=str(config.IMAGES_DIR)),
        name="images",
    )

    # Serve the frontend HTML at root
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_frontend():
        if FRONTEND_PATH.exists():
            return HTMLResponse(content=FRONTEND_PATH.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>frontend.html not found</h1>", status_code=404)

    # Include API routes
    app.include_router(router, tags=["Face Retrieval"])

    @app.on_event("startup")
    async def startup_event():
        """Initialize models, sync existing images, and start file watcher."""
        global _watcher
        print("=" * 60)
        print("  Face-Based Image Retrieval System")
        print("=" * 60)
        print(f"  Data directory:  {config.DATA_DIR}")
        print(f"  Images directory: {config.IMAGES_DIR}")
        print(f"  Database:        {config.DB_PATH}")
        print(f"  FAISS index:     {config.FAISS_INDEX_PATH}")
        print(f"  Model:           {config.MODEL_NAME}")
        print(f"  Similarity threshold: {config.SIMILARITY_THRESHOLD}")
        print(f"  Min face width:  {config.MIN_FACE_WIDTH}px")
        print("=" * 60)
        print("  Loading InsightFace models (first run downloads ~300MB)...")

        # Pre-load detector and models
        from app.detection.detector import FaceDetector
        FaceDetector.get_instance()
        print("  [OK] Models loaded successfully!")
        print(f"  Swagger UI: http://localhost:{config.API_PORT}/docs")
        print("=" * 60)

        # ── Auto-sync: index any images in images/ not yet in the database ──
        import threading
        def _sync_and_watch():
            _startup_sync()
            global _watcher
            _watcher = ImageWatcher(watch_dir=config.BASE_DIR / "images")
            _watcher.start()
            print("  [OK] Auto-watcher started — new photos will be indexed automatically.")

        threading.Thread(target=_sync_and_watch, daemon=True).start()

    @app.on_event("shutdown")
    async def shutdown_event():
        """Stop the file watcher gracefully."""
        global _watcher
        if _watcher:
            _watcher.stop()

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level="info",
    )
