## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-28 - [Matrix Normalization Optimization]
**Learning:** When computing cosine similarity in numpy (e.g., `MatryoshkaIndexer` reranking), it is significantly faster to compute the raw dot product first and scale the resulting 1D array by the pre-computed magnitudes (`dot(A, B) / (|A| * |B|)`), rather than allocating a large intermediate normalized matrix (`A/|A|`).
**Action:** Always prefer raw dot product and vector magnitude scaling over full matrix normalization for large numpy arrays to reduce memory allocation overhead and improve latency.
