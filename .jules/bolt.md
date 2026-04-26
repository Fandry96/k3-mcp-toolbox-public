## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2024-04-26 - Matrix Normalization in Hot Loop
**Learning:** In computing cosine similarity for reranking, computing a normalized matrix `A / np.linalg.norm(A, axis=1)` is extremely slow due to intermediate memory allocations. `np.linalg.norm` is also less optimal than using `np.einsum('ij,ij->i', A, A)`.
**Action:** Instead of normalizing the full subset matrix and the query independently, compute the raw dot product first `np.dot(m_subset, q_vec)`, and then scale the result by the pre-computed 1D norms.
