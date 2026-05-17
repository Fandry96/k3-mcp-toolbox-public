## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-03-05 - Avoid Full Matrix Normalization in Numpy Cosine Similarity
**Learning:** When calculating cosine similarity over large matrices in `MatryoshkaIndexer`, normalizing the entire NxD matrix (`A / np.linalg.norm(A)`) allocates a massive intermediate array and slows down execution.
**Action:** Always compute raw dot products and scale them by 1D squared norms (computed efficiently with `np.einsum('ij,ij->i', A, A)`). This prevents unnecessary O(N*D) memory allocations, offering up to a ~6x speedup.
