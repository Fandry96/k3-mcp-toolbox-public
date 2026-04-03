## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-04-03 - [Optimize Whitespace Normalization with split/join]
**Learning:** Using regex (`re.sub(r"\s+", " ", text).strip()`) for whitespace normalization is significantly slower than string methods. Benchmarking shows that `" ".join(text.split())` provides a ~5.4x speedup for this operation.
**Action:** When normalizing whitespace in hot paths (like text sanitization during indexing), always prefer `" ".join(text.split())` over regex substitution.
