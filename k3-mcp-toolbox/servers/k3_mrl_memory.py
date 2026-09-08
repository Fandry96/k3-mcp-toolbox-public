#!/usr/bin/env python3
"""
k3-mrl-memory — High-Speed Semantic Vector Search MCP Server
Exposes semantic search over the K3 25,000+ vector MRL Index (mrl_index.pkl).
Supports full corpus search, skill-specific search, and index telemetry.
"""

import os
import sys
import pickle
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import requests

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("k3-mrl-memory")
mcp._mcp_server.version = "1.0.0"

# Paths
DEFAULT_INDEX_PATH = Path(
    os.environ.get(
        "K3_MRL_INDEX_PATH",
        Path(__file__).resolve().parent.parent / "src" / "mrl_index.pkl"
    )
).resolve()
TARGET_DIMENSION = 768
MODEL = "models/gemini-embedding-001"

# In-memory index cache
_INDEX_CACHE: Optional[Dict[str, Any]] = None
_MATRIX_CACHE: Optional[np.ndarray] = None
_KEYS_CACHE: Optional[List[str]] = None
_LAST_LOAD_TIME: float = 0.0


def _get_api_key() -> str:
    """Retrieves the Gemini API key from environment variables."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
    return key


def _embed_query(query: str) -> np.ndarray:
    """Embeds a query using gemini-embedding-001 with MRL 768 truncation and L2 normalization."""
    api_key = _get_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:embedContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload = {
        "model": MODEL,
        "content": {"parts": [{"text": query[:8000]}]},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    r.raise_for_status()
    raw_vec = r.json()["embedding"]["values"]

    vec = np.array(raw_vec, dtype=np.float32)
    if TARGET_DIMENSION < vec.shape[0]:
        vec = vec[:TARGET_DIMENSION]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _ensure_index() -> tuple:
    """Loads and caches the MRL index into memory with precomputed matrix for O(N) top-k."""
    global _INDEX_CACHE, _MATRIX_CACHE, _KEYS_CACHE, _LAST_LOAD_TIME

    if _INDEX_CACHE is not None:
        return _INDEX_CACHE, _MATRIX_CACHE, _KEYS_CACHE

    if not DEFAULT_INDEX_PATH.exists():
        raise FileNotFoundError(f"MRL index not found at {DEFAULT_INDEX_PATH}")

    t0 = time.time()
    with open(DEFAULT_INDEX_PATH, "rb") as f:
        _INDEX_CACHE = pickle.load(f)

    # Pre-build normalized matrix and keys array for vector search
    keys = list(_INDEX_CACHE.keys())
    vectors = [
        _INDEX_CACHE[k]["vector"] if isinstance(_INDEX_CACHE[k]["vector"], np.ndarray)
        else np.array(_INDEX_CACHE[k]["vector"], dtype=np.float32)
        for k in keys
    ]
    matrix = np.vstack(vectors)

    # Ensure matrix is float32 and L2 normalized
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    _MATRIX_CACHE = (matrix / norms).astype(np.float32)
    _KEYS_CACHE = keys
    _LAST_LOAD_TIME = time.time() - t0

    sys.stderr.write(f"[k3-mrl-memory] Loaded {len(keys):,} vectors in {_LAST_LOAD_TIME:.2f}s\n")
    return _INDEX_CACHE, _MATRIX_CACHE, _KEYS_CACHE


def _classify_key(key: str) -> str:
    """Determines category from key path."""
    k_lower = key.lower()
    if "proto_book" in k_lower or "book" in k_lower:
        return "book"
    if "skill" in k_lower:
        return "skill"
    if "knowledge" in k_lower or "\\ki_" in k_lower:
        return "knowledge"
    if "brain" in k_lower:
        return "brain"
    if "research" in k_lower or "night" in k_lower or "deep" in k_lower:
        return "research"
    return "general"


@mcp.tool()
def mrl_search(query: str, top_k: int = 10, filter_type: Optional[str] = None) -> str:
    """
    Search the entire K3 semantic memory index (25,000+ chunks) via Matryoshka vector search.
    Args:
        query: Natural language query (e.g. 'governance lifecycle hooks', 'Delaware M&A earnout')
        top_k: Number of results to return (default: 10, max: 50)
        filter_type: Optional category filter: 'skill', 'knowledge', 'research', 'brain', 'book'
    """
    if not query or not query.strip():
        return "Error: Search query cannot be empty."

    try:
        index, matrix, keys = _ensure_index()
    except Exception as e:
        return f"Error loading index: {e}"

    try:
        q_vec = _embed_query(query)
    except Exception as e:
        return f"Error embedding query: {e}"

    top_k = min(max(1, top_k), 50)

    # Cosine similarity via dot product (both q_vec and matrix are normalized)
    scores = np.dot(matrix, q_vec)

    # Apply filter mask if requested
    if filter_type:
        filter_norm = filter_type.strip().lower()
        mask = np.array([_classify_key(k) == filter_norm for k in keys], dtype=bool)
        if not np.any(mask):
            return f"No records found matching filter '{filter_type}'."
        scores = np.where(mask, scores, -np.inf)

    # Efficient top-k partition
    actual_k = min(top_k, len(scores))
    if actual_k <= 0:
        return "Index is empty."

    partition_indices = np.argpartition(scores, -actual_k)[-actual_k:]
    sorted_indices = partition_indices[np.argsort(-scores[partition_indices])]

    results = []
    for rank, idx in enumerate(sorted_indices, 1):
        score = float(scores[idx])
        if score == -np.inf:
            continue
        key = keys[idx]
        item = index[key]
        snippet = item.get("snippet", "").strip().replace("\n", " ")[:250]
        cat = _classify_key(key)
        results.append(
            f"{rank}. [{cat.upper()}] (Score: {score:.4f}) {key}\n"
            f"   Snippet: {snippet}..."
        )

    if not results:
        return f"No results found for query: '{query}'"

    header = f"=== MRL Search Results ({len(results)} matches for: '{query}') ==="
    if filter_type:
        header += f" [Filter: {filter_type}]"
    return header + "\n\n" + "\n\n".join(results)


@mcp.tool()
def mrl_search_skills(query: str, top_k: int = 5) -> str:
    """
    Fast skill discovery: search exclusively across active SKILL.md files.
    Returns matched skills with relevance scores and file paths.
    """
    return mrl_search(query=query, top_k=top_k, filter_type="skill")


@mcp.tool()
def mrl_index_stats() -> str:
    """
    Returns telemetry and diagnostics for the MRL index: total vectors, memory footprint,
    dimension size, and breakdown across corpora categories.
    """
    try:
        index, matrix, keys = _ensure_index()
    except Exception as e:
        return f"Index unavailable: {e}"

    categories = {}
    for k in keys:
        cat = _classify_key(k)
        categories[cat] = categories.get(cat, 0) + 1

    file_size_mb = DEFAULT_INDEX_PATH.stat().st_size / (1024 * 1024)
    matrix_mb = matrix.nbytes / (1024 * 1024)

    lines = [
        "=== K3 MRL Memory Index Telemetry ===",
        f"Path: {DEFAULT_INDEX_PATH}",
        f"File Size: {file_size_mb:.1f} MB",
        f"Total Vectors: {len(keys):,}",
        f"Vector Dimension: {TARGET_DIMENSION} (MRL normalized)",
        f"Matrix RAM Cache: {matrix_mb:.1f} MB",
        f"Initial Load Time: {_LAST_LOAD_TIME:.2f}s",
        "\nCorpus Breakdown:",
    ]
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = (count / len(keys)) * 100
        lines.append(f"  - {cat.capitalize():<12}: {count:>6,} chunks ({pct:.1f}%)")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
