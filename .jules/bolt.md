## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-05-19 - Fast Reranking via np.einsum and 1D Norms
**Learning:** Computing cosine similarity in numpy (e.g. MatryoshkaIndexer reranking) by first normalizing an NxD matrix `A / np.linalg.norm(A)` causes a massive memory allocation on hot paths. Using `np.einsum('ij,ij->i', A, A)` to calculate 1D squared norms, computing the raw dot product `np.dot(A, B)`, and then scaling by the norms gives a ~4-5x speedup for 10K+ candidates.
**Action:** Always compute raw dot products and scale by 1D norms for matrix-vector cosine similarity instead of normalizing the entire matrix first.

## 2025-05-19 - String Split/Join vs Regex Whitespace Normalization
**Learning:** Using `re.sub(r"\s+", " ", text).strip()` for text sanitization is significantly slower than `" ".join(text.split())`. Benchmarking showed a ~4.6x speedup.
**Action:** Replace whitespace normalization regexes with string split/join in text processing hot paths.
