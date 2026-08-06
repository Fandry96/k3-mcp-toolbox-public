## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize High-Res Reranking]
**Learning:** In MRL Indexer's STAGE 2 High-Res Reranking, the intermediate NxD normalized matrix allocation (`m_full_subset / np.linalg.norm(...)`) causes unnecessary memory overhead and slows down scoring. Replacing it with raw dot products scaled by 1D squared norms via `np.einsum('ij,ij->i', A, A)` provides an ~3.4x speedup (e.g., from 1.3s down to 0.38s for 1000 items x 1000 iterations).
**Action:** When calculating cosine similarity in `k3_mrl_indexer.py`, use 1D squared norms (`np.einsum('ij,ij->i', A, A)`) and raw dot products instead of `np.linalg.norm` and matrix division.
