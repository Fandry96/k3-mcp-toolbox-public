## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2024-04-08 - [Optimize Whitespace Normalization]
**Learning:** Using regex (`re.sub(r"\s+", " ", text).strip()`) for whitespace normalization is significantly slower than string manipulation methods in Python. Benchmarks show ` " ".join(text.split())` is ~3x faster for text sanitization in hot paths.
**Action:** When sanitizing text, especially in high-throughput data processing or indexing pipelines, prefer `" ".join(text.split())` over regex for whitespace normalization.
