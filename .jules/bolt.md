## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.
## 2024-05-11 - [Optimize Stage 2 Cosine Similarity]
**Learning:** Using `np.linalg.norm` with `keepdims=True` followed by division allocates a full NxD intermediate matrix which is memory intensive and slower inside hot loops.
**Action:** Compute 1D squared norms with `np.einsum('ij,ij->i', m, m)` and scale the dot product directly to skip large intermediate allocations.
