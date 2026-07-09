## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.
## 2025-07-09 - Memory-Efficient Cosine Similarity
**Learning:** Computing cosine similarity in numpy by explicitly normalizing NxD matrices (`A / np.linalg.norm(A)`) causes a massive memory overhead and slowdown due to intermediate array allocation.
**Action:** Use raw dot products and scale them with 1D squared norms (`np.einsum('ij,ij->i', A, A)`) instead, achieving ~10x speedups for large matrices.
