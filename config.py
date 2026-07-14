from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── App ───────────────────────────────────────────────────────────
APP_NAME    = "EstateGPT"
APP_VERSION = "1.0.0"

# ── API Keys ─────────────────────────────────────────────────────
SCRAPEGRAPH_API_KEY = os.getenv("SCRAPEGRAPH_API_KEY", "")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY",        "")
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY",      "")   
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")   # Nearby places
NVIDIA_API_KEY      = os.getenv("NVIDIA_API_KEY",      "")   # Embeddings & reranker

# ── Model Names ───────────────────────────────────────────────────
GROQ_MODEL      = os.getenv("GROQ_MODEL",      "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# ── MongoDB ───────────────────────────────────────────────────────
MONGODB_URI              = os.getenv("MONGODB_URI",              "mongodb://localhost:27017")
MONGODB_DB               = os.getenv("MONGODB_DB",               "estategpt")
MONGODB_COLLECTION       = os.getenv("MONGODB_COLLECTION",       "properties")
MONGODB_CHUNK_COLLECTION = os.getenv("MONGODB_CHUNK_COLLECTION", "property_chunks")