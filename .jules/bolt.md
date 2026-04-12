## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-04-12 - [Optimize Cosine Similarity Calculation]
**Learning:** Using `np.linalg.norm` with `keepdims=True` and performing full matrix normalization `A / norm(A)` allocates massive intermediate matrices and performs poorly. Calculating raw dot products and scaling with 1D norms via `np.einsum('ij,ij->i', A, A)` is significantly faster (~6x in caching, ~2x in reranking) and reduces memory overhead.
**Action:** Always prefer computing raw dot products and scaling them with 1D norms via `np.einsum` over normalizing full matrices when calculating cosine similarity.
