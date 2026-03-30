"""
Test Script -- End-to-end demonstration.

Uploads all sample images, triggers clustering, then queries with
one image to demonstrate full retrieval pipeline.

Usage:
    1. Start the server:  python main.py
    2. Run this script:   python test_script.py
"""

import os
import sys
import time
import glob
import requests

# --- Configuration ---
BASE_URL = os.environ.get("FACE_API_URL", "http://localhost:8000")
# Use the images/ directory in the project root for sample images
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


def check_server():
    """Verify the API server is running."""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        print(f"[OK] Server is healthy: {data}")
        return True
    except Exception as e:
        print(f"[FAIL] Server not reachable at {BASE_URL}: {e}")
        print("  Start the server first: python main.py")
        return False


def upload_images(image_dir: str):
    """Upload all images from a directory."""
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for pattern in patterns:
        image_files.extend(glob.glob(os.path.join(image_dir, pattern)))

    if not image_files:
        print(f"[FAIL] No images found in {image_dir}")
        return []

    print(f"\n{'='*60}")
    print(f"  UPLOADING {len(image_files)} IMAGES")
    print(f"{'='*60}")

    results = []
    for i, img_path in enumerate(sorted(image_files), 1):
        filename = os.path.basename(img_path)
        print(f"\n  [{i}/{len(image_files)}] Uploading: {filename}")

        with open(img_path, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/upload",
                files={"file": (filename, f, "image/jpeg")},
                timeout=120,
            )

        if resp.status_code == 200:
            data = resp.json()
            print(f"    [OK] Image ID: {data['image_id']}")
            print(f"    [OK] Faces detected: {data['faces_detected']}, stored: {data['faces_stored']}, skipped: {data['faces_skipped']}")
            for face in data.get("faces", []):
                print(f"      -> Face {face['face_id'][:8]}... (score: {face['det_score']}, bbox: {face['bbox']})")
            results.append(data)
        else:
            print(f"    [FAIL] Upload failed: {resp.status_code} -- {resp.text}")

    return results


def run_clustering():
    """Trigger DBSCAN clustering."""
    print(f"\n{'='*60}")
    print(f"  CLUSTERING FACES")
    print(f"{'='*60}")

    resp = requests.post(f"{BASE_URL}/recluster", timeout=120)

    if resp.status_code == 200:
        data = resp.json()
        print(f"  [OK] Total faces: {data['total_faces']}")
        print(f"  [OK] Clusters found: {data['total_clusters']}")
        print(f"  [OK] Noise faces: {data['noise_faces']}")
        for cluster in data.get("clusters", []):
            print(f"    -> Cluster {cluster['cluster_id']}: {cluster['face_count']} faces across {cluster['image_count']} images")
        return data
    else:
        print(f"  [FAIL] Clustering failed: {resp.status_code} -- {resp.text}")
        return None


def query_face(image_path: str):
    """Query with a face image to find matching full images."""
    filename = os.path.basename(image_path)
    print(f"\n{'='*60}")
    print(f"  QUERYING: {filename}")
    print(f"{'='*60}")

    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/query",
            files={"file": (filename, f, "image/jpeg")},
            timeout=120,
        )

    if resp.status_code == 200:
        data = resp.json()
        print(f"  [OK] Matches found: {data['matches_found']}")
        print(f"  [OK] Best cluster: {data['cluster_id']}")
        print(f"  [OK] Total matching images: {data['total_images']}")

        if data.get("top_matches"):
            print(f"\n  Top matches:")
            for m in data["top_matches"]:
                print(f"    -> Face {m['face_id'][:8]}... (similarity: {m['similarity']})")

        if data.get("images"):
            print(f"\n  Matching images:")
            for img in data["images"]:
                print(f"    -> {img['image_id']}: {img['image_path']}")

        return data
    else:
        print(f"  [FAIL] Query failed: {resp.status_code} -- {resp.text}")
        return None


def get_cluster_results(cluster_id: int):
    """Get all images for a specific cluster."""
    print(f"\n{'='*60}")
    print(f"  CLUSTER {cluster_id} RESULTS")
    print(f"{'='*60}")

    resp = requests.get(f"{BASE_URL}/results/{cluster_id}", timeout=30)

    if resp.status_code == 200:
        data = resp.json()
        print(f"  [OK] Cluster {data['cluster_id']}: {data['face_count']} faces, {data['image_count']} images")
        for img in data.get("images", []):
            print(f"    -> {img['image_id']}: {img['image_path']}")
        return data
    else:
        print(f"  [FAIL] Failed: {resp.status_code} -- {resp.text}")
        return None


def main():
    """Run the full test pipeline."""
    print("\n" + "=" * 60)
    print("  FACE-BASED IMAGE RETRIEVAL -- TEST SCRIPT")
    print("=" * 60)

    # 1. Check server
    if not check_server():
        sys.exit(1)

    # 2. Upload all sample images
    if not os.path.isdir(SAMPLE_DIR):
        print(f"\n[FAIL] Sample directory not found: {SAMPLE_DIR}")
        print("  Place sample images in the 'images/' directory")
        sys.exit(1)

    upload_results = upload_images(SAMPLE_DIR)
    if not upload_results:
        print("\n[FAIL] No images were uploaded successfully")
        sys.exit(1)

    # 3. Run clustering
    cluster_data = run_clustering()

    # 4. Query with the first sample image
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    sample_images = []
    for pattern in patterns:
        sample_images.extend(glob.glob(os.path.join(SAMPLE_DIR, pattern)))

    if sample_images:
        query_image = sorted(sample_images)[0]
        query_result = query_face(query_image)

        # 5. If a cluster was found, get its full results
        if query_result and query_result.get("cluster_id") is not None:
            get_cluster_results(query_result["cluster_id"])

    print(f"\n{'='*60}")
    print("  TEST COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
