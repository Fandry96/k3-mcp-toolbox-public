## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.
## 2026-05-22 - [Cosine Similarity Optimization]
**Learning:** Intermediate NxD matrix allocation for normalization is a massive bottleneck in numpy cosine similarity. The 1D squared norms via `np.einsum` are up to ~10x faster.
**Action:** Always use raw dot products scaled by 1D squared norms (`np.einsum('ij,ij->i', A, A)`) instead of `A / np.linalg.norm(A)` on hot paths.
