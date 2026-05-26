## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2024-05-24 - [Avoid NxD intermediate allocations during on-the-fly normalization]
**Learning:** In high-dimensional vector reranking (e.g. 768 dims), allocating an intermediate NxD normalized matrix before computing cosine similarity is a bottleneck. Using a raw dot product scaled by 1D norms calculated via `np.einsum('ij,ij->i', A, A)` prevents massive memory allocation inside hot loops and reduces latency.
**Action:** Always compute dot products first and divide by norms for on-the-fly similarity scoring instead of pre-normalizing large submatrices.
