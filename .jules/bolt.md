## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize Cosine Similarity Normalization in NumPy]
**Learning:** When computing cosine similarity (`np.dot(A_norm, B_norm)`) in a hot loop with NumPy, pre-allocating intermediate normalized NxD matrices (e.g., `A / np.linalg.norm(A, axis=1)`) incurs massive memory allocation overhead. Additionally, `np.linalg.norm` is slow for computing 1D norms.
**Action:** Instead of normalizing the full matrix before dot product, compute the raw dot product first, and scale the result by the 1D norms. Use `np.einsum('ij,ij->i', A, A)` to compute squared 1D norms significantly faster than `np.linalg.norm`.
