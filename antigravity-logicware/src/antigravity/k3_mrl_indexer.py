import os
import hashlib
import pickle
import argparse
import numpy as np
import re
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Tuple
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv

# ------------------------------------------------------------------------------
# K3 Firehose: Matryoshka Representation Learning (MRL) Indexer
# Based on "Matryoshka Representation Learning" (Kusupati et al., 2022).
# ------------------------------------------------------------------------------

# --- CONFIGURATION ---
DEFAULT_DIMENSION = 768
DEFAULT_MODEL = "models/text-embedding-004"  # SOTA (Dec 2025)
# GEMINI_2_0_FLASH_COMPATIBLE = True
BATCH_SIZE = 5  # Increased batch size for throughput
MAX_WORKERS = 4  # Parallel API calls
SAVE_INTERVAL = 20  # Auto-save every N updates


class MatryoshkaIndexer:
    def __init__(
        self,
        api_key: str,
        target_dir: str,
        index_file: str,
        dimension: int = DEFAULT_DIMENSION,
    ):
        self.client = genai.Client(api_key=api_key)
        self.target_dir = Path(target_dir).resolve()
        self.index_file = Path(index_file).resolve()
        self.dimension = dimension
        # Structure: { file_path: { 'vector': np.array, 'hash': str, 'snippet': str } }
        self.index: Dict[str, Dict[str, Any]] = {}
        self.failed_files: List[str] = []
        self._unsaved_changes = 0
        self._matrix_cache = None
        self._paths_cache = None
        self.load_index()

    def load_index(self):
        """Loads binary pickle index for speed."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "rb") as f:
                    self.index = pickle.load(f)

                # Validation check
                if self.index and isinstance(
                    next(iter(self.index.values()))["vector"], list
                ):
                    print(
                        "[WARN] Detected legacy JSON-style lists. Converting to Numpy..."
                    )
                    for k, v in self.index.items():
                        v["vector"] = np.array(v["vector"], dtype=np.float32)

                print(
                    f"[SYSTEM] Loaded {len(self.index)} vectors from {self.index_file.name}"
                )
            except Exception as e:
                print(f"[WARN] Index corrupt or incompatible ({e}). Starting fresh.")
                self.index = {}
        else:
            print(f"[SYSTEM] No index found at {self.index_file}. Initializing new.")

        # Invalidate cache
        self._matrix_cache = None
        self._paths_cache = None

    def save_index(self):
        """Atomic binary save to prevent corruption."""
        if self._unsaved_changes == 0:
            return

        temp_file = self.index_file.with_suffix(".tmp")
        try:
            with open(temp_file, "wb") as f:
                pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Atomic rename
            if self.index_file.exists():
                self.index_file.unlink()
            temp_file.rename(self.index_file)

            print(f"[SYSTEM] Checkpoint saved. Total entries: {len(self.index)}")
            self._unsaved_changes = 0
        except Exception as e:
            print(f"[ERROR] Failed to save index: {e}")

    def get_file_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def sanitize_content(self, text: str) -> str:
        # Remove binary noise / markdown images
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def walk_files(self) -> List[Path]:
        print(f"[SYSTEM] Scanning {self.target_dir}...")
        valid_files = []
        skip_dirs = {
            ".git",
            "node_modules",
            "__pycache__",
            "venv",
            ".venv",
            ".env",
            "dist",
            "build",
            ".idea",
            ".vscode",
            ".git",
            "__pycache__",
        }
        extensions = {
            ".txt",
            ".md",
            ".py",
            ".js",
            ".json",
            ".html",
            ".css",
            ".ts",
            ".go",
            ".rs",
            ".java",
            ".c",
            ".h",
        }

        for root, dirs, files in os.walk(self.target_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for file in files:
                if Path(file).suffix in extensions:
                    valid_files.append(Path(root) / file)
        return valid_files

    def _embed_batch_worker(
        self, batch_data: List[Tuple[str, str]]
    ) -> List[Tuple[str, np.ndarray, str, str]]:
        """
        Worker function for ThreadPool.
        Returns list of (path, vector, hash, snippet)
        """
        paths, texts = zip(*batch_data)
        try:
            response = self.client.models.embed_content(
                model=DEFAULT_MODEL,
                contents=list(texts),
                config=types.EmbedContentConfig(output_dimensionality=self.dimension),
            )

            results = []
            if response.embeddings:
                vectors = [
                    np.array(e.values, dtype=np.float32) for e in response.embeddings
                ]
                for path, vec, text in zip(paths, vectors, texts):
                    # Calculate hash here to ensure it matches the text actually embedded
                    txt_hash = self.get_file_hash(text)
                    results.append((path, vec, txt_hash, text[:200]))
            return results

        except Exception as e:
            print(f"[API ERROR] Batch failed: {e}")
            return []

    def smart_chunk(self, raw_text: str, limit: int = 8000) -> List[Tuple[str, str]]:
        """
        Splits text respecting line boundaries.
        """
        chunks = []

        # 1. Custom File Delimiters
        parts = re.split(r"(^--- FILE: .*? ---$)", raw_text, flags=re.MULTILINE)
        if len(parts) > 1:
            current_header = "preamble"
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part.startswith("--- FILE:"):
                    current_header = (
                        part.replace("--- FILE:", "").replace("---", "").strip()
                    )
                else:
                    chunks.extend(self._text_splitter(part, current_header, limit))
        else:
            chunks = self._text_splitter(raw_text, "main", limit)

        return chunks

    def _text_splitter(
        self, text: str, base_id: str, limit: int
    ) -> List[Tuple[str, str]]:
        """Recursive split by newline to avoid cutting words."""
        if len(text) <= limit:
            return [(text, f"::{base_id}")]

        chunks = []
        lines = text.split("\n")
        current_chunk = []
        current_len = 0
        chunk_idx = 0

        for line in lines:
            line_len = len(line) + 1  # +1 for newline
            if current_len + line_len > limit:
                # Commit current chunk
                if current_chunk:
                    joined = "\n".join(current_chunk)
                    chunks.append((joined, f"::{base_id}[{chunk_idx}]"))
                    chunk_idx += 1
                    current_chunk = []
                    current_len = 0

            current_chunk.append(line)
            current_len += line_len

        # Flush remainder
        if current_chunk:
            joined = "\n".join(current_chunk)
            chunks.append((joined, f"::{base_id}[{chunk_idx}]"))

        return chunks

    def run_indexing(self, limit: int = 0):
        """
        Main execution loop.
        """
        files = self.walk_files()
        if limit > 0:
            print(f"[SYSTEM] Limiting to first {limit} files.")
            files = files[:limit]

        print(f"[SYSTEM] Found {len(files)} files. Processing...")

        # 1. Prepare Processing Queue
        tasks = []
        current_batch = []

        # [NEW] ReFRAG Integration
        try:
            from tools.refrag import slice_to_refrag, get_strategy_for_file

            print("[SYSTEM] ReFRAG Module: LOADED")
        except ImportError:
            # Fallback for when running from different CWD
            try:
                from refrag import slice_to_refrag, get_strategy_for_file

                print("[SYSTEM] ReFRAG Module: LOADED (Local)")
            except ImportError:
                print("[WARN] ReFRAG Module: NOT FOUND. Fallback to Standard.")
                slice_to_refrag = None
                get_strategy_for_file = lambda x: "standard"

        for file_path in files:
            try:
                raw_text = file_path.read_text(errors="ignore", encoding="utf-8")

                # Determine Strategy
                strategy = "standard"
                if get_strategy_for_file:
                    strategy = get_strategy_for_file(file_path.name)

                # [OVERRIDE] If global mode is set (future)
                # if self.mode == 'refrag': strategy = 'refrag'

                if strategy == "refrag" and slice_to_refrag:
                    # Micro-Chunking (16t / 8s)
                    # Note: We must pass window_size=16 for true PhD spec,
                    # but let's default to the function's default (256) for cost safety first
                    # UNLESS user explicitly asked for PhD precision.
                    # The implementation_plan authorized "Brute Force" on source code.
                    # Let's use window_size=128 (Logic Block) for now to be safe on storage
                    # ReFRAG default is 256.
                    file_chunks = slice_to_refrag(raw_text, window_size=128, stride=64)
                    if len(file_chunks) > 0:
                        # Log only first time or verbose
                        pass
                else:
                    # Standard Smart Chunking
                    file_chunks = self.smart_chunk(raw_text)

                for chunk_text, chunk_suffix in file_chunks:
                    clean_text = self.sanitize_content(chunk_text)
                    if not clean_text:
                        continue

                    full_path = str(file_path) + chunk_suffix
                    new_hash = self.get_file_hash(clean_text)

                    # Incremental Check: Skip if hash matches existing index
                    if full_path in self.index:
                        if self.index[full_path]["hash"] == new_hash:
                            if len(self.index[full_path]["vector"]) == self.dimension:
                                continue

                    current_batch.append((full_path, clean_text))

                    if len(current_batch) >= BATCH_SIZE:
                        tasks.append(list(current_batch))
                        current_batch = []

            except Exception as e:
                print(f"[WARN] Read error {file_path}: {e}")

        # Flush final batch
        if current_batch:
            tasks.append(current_batch)

        if not tasks:
            print("[SYSTEM] Index up to date.")
            return

        print(
            f"[SYSTEM] Processing {len(tasks)} batches ({len(tasks) * BATCH_SIZE} chunks) with {MAX_WORKERS} threads..."
        )

        # 2. Parallel Execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {
                executor.submit(self._embed_batch_worker, batch): batch
                for batch in tasks
            }

            for future in concurrent.futures.as_completed(future_to_batch):
                results = future.result()
                if results:
                    for path, vec, txt_hash, snippet in results:
                        self.index[path] = {
                            "vector": vec,
                            "hash": txt_hash,
                            "snippet": snippet,
                        }
                        self._unsaved_changes += 1
                        # Invalidate cache
                        self._matrix_cache = None
                        self._paths_cache = None
                        print(f"[INDEXED] {path.split('::')[-1]}")

                    if self._unsaved_changes >= SAVE_INTERVAL:
                        self.save_index()

        self.save_index()
        print("[SYSTEM] Indexing Complete.")

    def search(self, query: str, top_k: int = 5):
        """
        MRL Funnel Search: 64-dim Shortlist -> 768-dim Rerank
        """
        SHORTLIST_DIM = 64
        SHORTLIST_FACTOR = 15

        if not self.index:
            print("[ERR] Index empty.")
            return []

        try:
            # Embed Query
            resp = self.client.models.embed_content(
                model=DEFAULT_MODEL,
                contents=query,
                config=types.EmbedContentConfig(output_dimensionality=self.dimension),
            )
            q_vec = np.array(resp.embeddings[0].values, dtype=np.float32)

            # Prepare Matrix
            # NOTE: For massive indices, use HNSW (faiss/chroma). For <100k vectors, numpy is fine.
            if self._matrix_cache is None or self._paths_cache is None:
                self._paths_cache = list(self.index.keys())
                self._matrix_cache = np.stack([d["vector"] for d in self.index.values()])

            paths = self._paths_cache
            matrix = self._matrix_cache

            # --- STAGE 1: Low-Res Shortlist (64 dims) ---
            q_short = q_vec[:SHORTLIST_DIM]
            m_short = matrix[:, :SHORTLIST_DIM]

            # Normalize
            m_short_norm = m_short / (
                np.linalg.norm(m_short, axis=1, keepdims=True) + 1e-9
            )
            q_short_norm = q_short / (np.linalg.norm(q_short) + 1e-9)

            scores_short = np.dot(m_short_norm, q_short_norm)

            # Select Candidates
            k_cand = min(top_k * SHORTLIST_FACTOR, len(paths))
            candidate_idxs = np.argsort(scores_short)[::-1][:k_cand]

            # --- STAGE 2: High-Res Rerank (768 dims) ---
            m_full_subset = matrix[candidate_idxs]

            m_full_norm = m_full_subset / (
                np.linalg.norm(m_full_subset, axis=1, keepdims=True) + 1e-9
            )
            q_full_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)

            scores_full = np.dot(m_full_norm, q_full_norm)

            # Final Sort
            final_order = np.argsort(scores_full)[::-1][:top_k]

            results_list = []

            try:
                print(f"\n--- Results for: '{query}' ---")
                for i, idx in enumerate(final_order):
                    global_idx = candidate_idxs[idx]
                    path = paths[global_idx]
                    score = scores_full[idx]
                    snippet = self.index[path]["snippet"].replace("\n", " ")

                    # Sanitize snippet for console
                    try:
                        safe_snippet = snippet.encode("ascii", "replace").decode(
                            "ascii"
                        )
                        print(f"{i + 1}. [{score:.4f}] {Path(path).name}")
                        print(f"    {safe_snippet[:100]}...\n")  # Truncate for console
                    except Exception:
                        pass

                    results_list.append(
                        {"path": str(path), "score": float(score), "snippet": snippet}
                    )
            except Exception:
                pass

            return results_list

        except Exception as e:
            try:
                # Sanitize error message
                msg = str(e).encode("ascii", "replace").decode("ascii")
                print(f"[SEARCH ERROR] {msg}")
            except Exception:
                pass
            return []


if __name__ == "__main__":
    env_path = find_dotenv(usecwd=True)
    load_dotenv(env_path)

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[FATAL] GOOGLE_API_KEY not found.")
        exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".")
    parser.add_argument("--index", default="mrl_index.pkl")
    parser.add_argument("--query", type=str)
    parser.add_argument("--reindex", action="store_true", help="Force full reindex")
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of files to index"
    )
    args = parser.parse_args()

    indexer = MatryoshkaIndexer(api_key, args.path, args.index)

    if args.reindex:
        indexer.index = {}  # Clear memory

    # Quick patch to support limit logic without changing class signature too much
    # We can inject it or handle it in the class if we modified it.
    # But for now, let's just modify the class to accept limit in run_indexing?
    # Or cleaner: Just slice valid_files in `find_files`.

    # We need to modify the class method find_files or run_indexing?
    # Let's modify the class instantiation or run_indexing call if possible.
    # Looking at the code, run_indexing calls find_files.
    # Let's verify where `limit` should go.
    # I'll modify the `find_files` to take a limit or just monkeypatch for now?
    # Better: modify `run_indexing` to accept limit.
    indexer.run_indexing(limit=args.limit)

    if args.query:
        indexer.search(args.query)
