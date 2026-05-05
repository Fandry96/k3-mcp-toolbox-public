## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-24 - [Optimize Whitespace Normalization with Split/Join]
**Learning:** In hot paths like `sanitize_content` inside the MRL Indexer, using `re.sub(r"\s+", " ", text).strip()` to normalize whitespace is surprisingly slow compared to pure string methods. Benchmarks show that `" ".join(text.split())` is approximately 4.95x faster while achieving the exact same result (collapsing all whitespace characters).
**Action:** Replace `re.sub` for whitespace normalization with `" ".join(text.split())` in text processing pipelines to reduce CPU overhead.
