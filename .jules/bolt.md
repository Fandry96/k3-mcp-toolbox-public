## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-03-03 - [Optimize Whitespace Normalization in Text Sanitization]
**Learning:** `re.sub(r"\s+", " ", text).strip()` is significantly slower (~5x slower in benchmarks) than ` " ".join(text.split())` for normalizing multiple whitespace characters into single spaces. Since `sanitize_content` is called iteratively across all chunks in the codebase during MRL indexing, string method replacements offer a substantial throughput improvement over regex on large codebases.
**Action:** Always prefer `" ".join(text.split())` over regex for generic whitespace normalization and trimming in hot paths.
