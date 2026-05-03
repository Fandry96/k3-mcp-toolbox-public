## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - Avoid intermediate full normalization matrix allocation
**Learning:** In MatryoshkaIndexer's Stage 2 reranking, allocating an intermediate normalized N x D matrix (`A / np.linalg.norm(A)`) causes a significant slow down in hot loops compared to computing a raw dot product directly and scaling by 1D squared norms (computed via `np.einsum`).
**Action:** Use `np.einsum('ij,ij->i', m, m)` to compute 1D squared norms and scale the raw dot product directly instead of using `np.linalg.norm` and full matrix scaling.
