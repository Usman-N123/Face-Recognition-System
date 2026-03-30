"""
Configuration module for Face-Based Image Retrieval System.
All paths, thresholds, and constants are centralized here.
Override any value via environment variables.
"""

import os
from pathlib import Path

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(os.environ.get("FACE_APP_BASE_DIR", Path(__file__).parent))
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "database.db"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
FAISS_ID_MAP_PATH = DATA_DIR / "faiss_id_map.json"

# Create directories on import
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ─── InsightFace Model ────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("FACE_MODEL_NAME", "buffalo_l")
DET_SIZE = (640, 640)  # Detection input resolution
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# ─── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 512  # ArcFace output dimensionality

# ─── Similarity Thresholds ────────────────────────────────────────────────────
# Cosine similarity: 1.0 = identical, 0.0 = orthogonal
# Lowered from 0.25 → 0.20 so side-angle / partially-lit faces are matched
SIMILARITY_THRESHOLD = float(os.environ.get("FACE_SIM_THRESHOLD", "0.20"))
RELAXED_THRESHOLD = float(os.environ.get("FACE_RELAXED_THRESHOLD", "0.35"))

# ─── Quality Control ──────────────────────────────────────────────────────────
# Lowered: WhatsApp-compressed photos are often small / somewhat blurry
MIN_FACE_WIDTH = int(os.environ.get("FACE_MIN_WIDTH", "25"))
BLUR_THRESHOLD = float(os.environ.get("FACE_BLUR_THRESHOLD", "8.0"))
MIN_DET_SCORE = float(os.environ.get("FACE_MIN_DET_SCORE", "0.3"))

# ─── FAISS Search ─────────────────────────────────────────────────────────────
# Increased top-K so we have more candidates for cluster voting
FAISS_TOP_K = int(os.environ.get("FACE_TOP_K", "10000"))

# ─── DBSCAN Clustering ────────────────────────────────────────────────────────
# Widened eps (0.5 → 0.65): allows more pose/lighting variation per cluster
DBSCAN_EPS = float(os.environ.get("FACE_DBSCAN_EPS", "0.65"))
DBSCAN_MIN_SAMPLES = int(os.environ.get("FACE_DBSCAN_MIN_SAMPLES", "1"))

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST = os.environ.get("FACE_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("FACE_API_PORT", "8000"))
