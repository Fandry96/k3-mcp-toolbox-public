#!/usr/bin/env python3
"""
k3-mrl-memory — High-Speed Semantic Vector Search MCP Server
Exposes semantic search over the K3 ~20,000-vector MRL Index (mrl_index.pkl).
Supports past-work memory search, skill discovery, and index telemetry.
"""

import datetime
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
DEFAULT_EXCLUDED_TYPES = {"tool", "skill", "rules"}

# In-memory index cache
_INDEX_CACHE: Optional[Dict[str, Any]] = None
_MATRIX_CACHE: Optional[np.ndarray] = None
_KEYS_CACHE: Optional[List[str]] = None
_LAST_LOAD_TIME: float = 0.0
_LAST_MTIME: float = 0.0


def _get_api_key() -> str:
    """Retrieves the Gemini API key from environment, .env.local, or Secret Manager."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    env_path = r"c:\K3_Firehose\.env.local"
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY=") or line.startswith("GOOGLE_API_KEY="):
                    return line.split("=", 1)[1].strip("'\"")
    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        name = "projects/gen-lang-client-0778100894/secrets/GEMINI_API_KEY/versions/latest"
        resp = client.access_secret_version(request={"name": name})
        return resp.payload.data.decode("utf-8").strip()
    except Exception:
        pass
    raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment, .env.local, or Secret Manager.")


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
    global _INDEX_CACHE, _MATRIX_CACHE, _KEYS_CACHE, _LAST_LOAD_TIME, _LAST_MTIME

    if not DEFAULT_INDEX_PATH.exists():
        raise FileNotFoundError(f"MRL index not found at {DEFAULT_INDEX_PATH}")

    current_mtime = DEFAULT_INDEX_PATH.stat().st_mtime
    if _INDEX_CACHE is not None and current_mtime == _LAST_MTIME:
        return _INDEX_CACHE, _MATRIX_CACHE, _KEYS_CACHE

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
    _LAST_MTIME = current_mtime

    sys.stderr.write(f"[k3-mrl-memory] Loaded {len(keys):,} vectors in {_LAST_LOAD_TIME:.2f}s\n")
    return _INDEX_CACHE, _MATRIX_CACHE, _KEYS_CACHE


def _classify_key(key: str) -> str:
    """Determines category from explicit key prefix first, then stripped file extension / path segments."""
    base = key.split("::", 1)[0].strip().lower().replace("\\", "/")

    # 1. Explicit prefixes first
    if base.startswith("mcp:"):
        return "tool"
    if base.startswith("skill:") or "/.agent/skills/" in base or "/skills/" in base:
        return "skill"
    if base.startswith("brain:"):
        return "brain"
    if base.startswith(("deep-research:", "deep-result:", "night3:", "nlm:", "tessa-manda:")):
        return "research"
    if (
        base.startswith("ki:k3-root/")
        or base.startswith("governance:")
        or ".agents/rules/" in base
        or base.startswith("proto-agent:rules/")
    ):
        if any(seg in base for seg in ("/memory.md", "/pitfalls.md", "/blueprints/", "/docs/")):
            return "knowledge"
        return "rules"
    if base.startswith("ki:"):
        return "knowledge"
    if base.startswith((
        "proto-agent:",
        "proto-script:",
        "proto-book:",
        "forge-ref:",
        "extraction:",
        "vault:",
        "template:",
    )):
        return "book"

    # 2. Stripped file extension check (after ::main / ::chunk[n] removal)
    if base.endswith(".docx"):
        return "research"

    # 3. Path-segment / substring fallbacks for unprefixed keys
    if "/knowledge/" in base or "/ki/" in base:
        return "knowledge"
    if "/brain/" in base:
        return "brain"
    if "proto_book" in base:
        return "book"
    if "research" in base or "night" in base or "deep" in base or "florida" in base:
        return "research"
    return "general"


def _parse_mtime_date(mtime_val: Any) -> Optional[datetime.date]:
    """Parses an entry's mtime value (ISO date/datetime string, epoch float/int, or date) into a datetime.date."""
    if mtime_val is None:
        return None
    if isinstance(mtime_val, datetime.datetime):
        return mtime_val.date()
    if isinstance(mtime_val, datetime.date):
        return mtime_val
    if isinstance(mtime_val, (int, float)):
        try:
            return datetime.date.fromtimestamp(float(mtime_val))
        except Exception:
            return None
    if isinstance(mtime_val, str):
        s = mtime_val.strip()
        if len(s) >= 10:
            try:
                return datetime.date.fromisoformat(s[:10])
            except ValueError:
                return None
    return None


def _format_mtime(mtime_val: Any) -> Optional[str]:
    """Formats an entry's mtime value as YYYY-MM-DD if valid."""
    d = _parse_mtime_date(mtime_val)
    return d.isoformat() if d is not None else None


@mcp.tool()
def mrl_search(
    query: str,
    top_k: int = 10,
    filter_type: Optional[str] = None,
    max_age_days: Optional[int] = None,
    include_all: bool = False,
) -> str:
    """
    Search the K3 semantic memory index (~20K chunks) for past work, decisions, pitfalls, and research.
    Use INSTEAD OF grep/view_file across brain/ and knowledge/ when the user asks whether something was solved, decided, or researched before.
    Do NOT use for: current code AST or call graphs (use graphify), live workspace files (use view_file), or skill lookup (use mrl_search_skills).
    By default (filter_type=None, include_all=False), excludes 'tool', 'skill', and 'rules' so results return actual past work ('brain', 'knowledge', 'research', 'book').
    Args:
        query: Natural language query (e.g. 'MCP port conflict zombie processes', 'Delaware M&A earnout')
        top_k: Number of results to return (default: 10, max: 50)
        filter_type: Optional category filter: 'brain', 'knowledge', 'research', 'book', 'rules', 'skill', 'tool', 'general', or 'all'
        max_age_days: Optional maximum age in days based on source file mtime (e.g. 30)
        include_all: If True (or filter_type='all'), search all categories including 'tool', 'skill', and 'rules'
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

    # Apply category filter mask
    filter_norm = filter_type.strip().lower() if filter_type else None
    if filter_norm and filter_norm not in ("all", "*"):
        mask = np.array([_classify_key(k) == filter_norm for k in keys], dtype=bool)
        if not np.any(mask):
            return f"No records found matching filter '{filter_type}'."
    elif not include_all and filter_norm not in ("all", "*"):
        mask = np.array([_classify_key(k) not in DEFAULT_EXCLUDED_TYPES for k in keys], dtype=bool)
    else:
        mask = np.ones(len(keys), dtype=bool)

    # Apply optional recency filter (max_age_days)
    if max_age_days is not None:
        cutoff = datetime.date.today() - datetime.timedelta(days=max(0, int(max_age_days)))
        age_mask = np.array(
            [
                (_d := _parse_mtime_date(index[k].get("mtime"))) is not None and _d >= cutoff
                for k in keys
            ],
            dtype=bool,
        )
        mask = mask & age_mask
        if not np.any(mask):
            return f"No results found for query: '{query}' within max_age_days={max_age_days}."

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
        mtime_str = _format_mtime(item.get("mtime"))
        mtime_tag = f" [mtime: {mtime_str}]" if mtime_str else ""
        results.append(
            f"{rank}. [{cat.upper()}]{mtime_tag} (Score: {score:.4f}) {key}\n"
            f"   Snippet: {snippet}..."
        )

    if not results:
        return f"No results found for query: '{query}'"

    header = f"=== MRL Search Results ({len(results)} matches for: '{query}') ==="
    if filter_type:
        header += f" [Filter: {filter_type}]"
    if max_age_days is not None:
        header += f" [MaxAge: {max_age_days}d]"
    return header + "\n\n" + "\n\n".join(results)


@mcp.tool()
def mrl_search_skills(query: str, top_k: int = 5) -> str:
    """
    Search exclusively across active SKILL.md cards in the ~20K MRL index to discover the right K3 skill for a task.
    Use INSTEAD OF manually browsing .agent/skills/ when looking up workflow or domain skills.
    Do NOT use for: past conversation recall (use mrl_search) or current code structure (use graphify).
    """
    return mrl_search(query=query, top_k=top_k, filter_type="skill")


@mcp.tool()
def mrl_index_stats() -> str:
    """
    Return telemetry and category breakdown for the ~20K-vector MRL index (total vectors, memory footprint, dimensions, and per-category counts).
    Use INSTEAD OF manually inspecting mrl_index.pkl when checking index health or corpus distribution.
    Do NOT use for: searching memory content (use mrl_search) or code graph stats (use graphify.graph_stats).
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

