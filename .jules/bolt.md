## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize String Whitespace Sanitization]
**Learning:** `re.sub(r"\s+", " ", text).strip()` is a common but slow pattern for removing extra whitespace. Using `" ".join(text.split())` is significantly faster. In benchmarks with a highly whitespace-polluted string repeated 1000 times, split/join took ~0.608s compared to ~3.344s for the regex method (~5.5x speedup). This pattern is present in the hot path for text chunking/indexing (e.g. `k3_mrl_indexer.py`).
**Action:** When normalizing whitespace in strings across the codebase, strictly prefer `" ".join(text.split())` over `re.sub(r"\s+", " ", text).strip()`.
