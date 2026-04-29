## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize Cosine Similarity Calculation with Einsum]
**Learning:** When calculating cosine similarity across many vectors in NumPy, normalizing an entire NxD matrix (`A / np.linalg.norm(A, axis=1, keepdims=True)`) creates a massive intermediate NxD allocation and is surprisingly slow.
**Action:** For large cosine similarity comparisons, compute the raw dot product first (`np.dot(A, B)`), compute 1D squared norms using `np.einsum('ij,ij->i', A, A)`, and divide the scalar scores. This avoids allocating a new NxD matrix and provides a ~3-5x speedup.
