# Face Recognition System

A production-ready, highly accurate Face-Based Image Retrieval System. The system automatically indexes photos dropped into a watch folder and allows users to find all their photos (including group photos) by matching their face via a webcam capture or an uploaded image.

## Overview

- **High Precision Face Recognition**: Uses `insightface` (ArcFace/RetinaFace) to detect and extract face embeddings.
- **Lightning-fast Vector Search**: Employs `FAISS` to instantly search through thousands of indexed faces.
- **Fully Automated Indexing**: A background worker powered by `watchdog` automatically detects new photos dropped into the `images/` folder and indexes them.
- **Sleek Web Interface**: A modern, dark-themed UI that supports both live webcam capture and direct photo uploads.
- **Clustering**: Groups faces to surface the most relevant matches across different photos dynamically.

## Prerequisites

- Python 3.9+
- A C++ compiler toolchain (required for FAISS and hnswlib)
- A webcam (optional, for face capture directly from the frontend UI)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Usman-N123/Face-Recognition-System.git
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On first run, the system will automatically download the required InsightFace models (~300MB).*

## Usage

1. **Start the Application:**
   Run the FastAPI server from the root directory:
   ```bash
   python main.py
   ```
   The backend and UI will spin up locally at `http://localhost:8000`.

2. **Add Source Photos:**
   Simply drop guest photos (`.jpg`, `.jpeg`, `.png`, `.jfif`) into the `images/` folder. The `watchdog` process running in the background will automatically detect and vector-index them in real time.

3. **Find Your Photos:**
   Navigate to `http://localhost:8000` in your web browser. 
   - Choose **Open Camera** to snap a quick photo of your face, or select **Upload** to provide a reference image from your computer.
   - Click **Find My Images** to retrieve a gallery of all photos containing your face.

## Project Architecture

- `app/`: Contains the FastAPI application logic and internal modules.
  - `api/`: REST endpoints (e.g., query logic, health checks, reclustering).
  - `clustering/`: DBSCAN clustering algorithms to manage identity vectors over time.
  - `database/`: SQLite database persistence for FAISS ids and associated image metadata.
  - `detection/`: InsightFace model wrapper for face coordinate and embedding generation.
  - `watcher/`: Background thread monitoring the source directory for real-time auto-indexing.
- `images/`: The source folder for fresh photos.
- `data/`: (Auto-generated) Stores the SQLite database (`faces.sqlite`), local `.faiss` indices, and state registries.
- `frontend.html`: Single-page modern user interface.
- `main.py`: Main application entry point uniting the API and background workers.

## License

This project is licensed under a **Custom Non-Commercial License**.

### You are allowed to:

* Use the code for personal and educational purposes
* Modify and distribute the code
* Share it with others (with proper attribution)

### You are NOT allowed to:

* Use this code for commercial purposes
* Sell this project or any derivative work
* Use it in any product or service that generates revenue

### Attribution Required

You must give appropriate credit to: **Muhammad Usman**

### Commercial Use

If you want to use this project commercially, you must obtain permission.

Contact: [usmannaeem350@gmail.com](mailto:usmannaeem350@gmail.com)

---

> ⚠️ This is **not an open-source license** as it restricts commercial use.

