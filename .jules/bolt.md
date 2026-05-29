## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-03-08 - Optimize Cosine Similarity Matrix Normalization
**Learning:** Intermediate NxD normalized matrix allocations (`A / np.linalg.norm(A)`) in hot paths like MatryoshkaIndexer search introduce significant memory overhead and slowdowns. Benchmarking demonstrated a ~3.35x speedup when these are replaced by computing the raw dot product and scaling by 1D squared norms.
**Action:** For large-scale cosine similarity computations, always use raw dot products scaled by `np.einsum('ij,ij->i', A, A)` instead of full matrix normalization to minimize memory allocations and improve throughput.
