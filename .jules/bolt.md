## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize Rerank Matrix Normalization]
**Learning:** Normalizing a full NxD matrix (e.g., `A / np.linalg.norm(A, axis=1)`) inside a hot loop is extremely slow and memory-intensive due to large array allocations.
**Action:** Compute the raw dot product first (`np.dot(A, B)`), and scale the result by the 1D norms instead (`np.sqrt(np.einsum('ij,ij->i', A, A))`). In benchmarks, this reduced Rerank normalization time from ~300ms to ~44ms for 100K candidates.
