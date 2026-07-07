## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize Cosine Similarity with raw dot products and 1D norms]
**Learning:** When computing cosine similarity in numpy (e.g., `MatryoshkaIndexer` scoring), avoiding intermediate normalized N x D matrix allocations (`A / np.linalg.norm(A)`) and using raw dot products scaled by 1D squared norms (`np.einsum('ij,ij->i', A, A)`) provides significant speedups. In benchmarks, it reduced STAGE 2 reranking time by ~4x-8x and dramatically reduced memory footprint.
**Action:** When calculating similarity between vectors, calculate raw dot products first and scale by 1D squared norms, rather than eagerly normalizing massive matrices.
