## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2024-05-18 - Avoid NxD explicit normalizations in hot loops
**Learning:** Benchmarking `MatryoshkaIndexer` revealed that calculating cosine similarity using intermediate normalized matrices (`m_short / np.linalg.norm(...)`) allocates massive NxD arrays repeatedly, creating significant memory and compute overhead.
**Action:** Replaced explicit normalization arrays with raw dot products scaled by 1D squared norms computed via `np.einsum('ij,ij->i', A, A)`. This avoids NxD matrix allocations and yields up to ~8.5x speedup for high-res reranking.
