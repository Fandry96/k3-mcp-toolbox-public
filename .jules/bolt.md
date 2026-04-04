## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-10-24 - [Optimize Whitespace Normalization]
**Learning:** Replacing regex-based whitespace normalization (`re.sub(r"\s+", " ", text).strip()`) with string split/join (`" ".join(text.split())`) provides a significant speedup (measured at ~5.45x in benchmarking). This is a highly recommended optimization target for `sanitize_content` hot paths across the codebase.
**Action:** When normalizing whitespace in text processing pipelines (especially inner loops like text chunk sanitization), always prioritize Python's built-in `str.split()` and `str.join()` over the `re` module unless complex pattern matching is strictly required.
