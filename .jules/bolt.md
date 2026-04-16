## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-04-16 - [Optimize whitespace normalization with string split/join]
**Learning:** For replacing multiple whitespace characters with a single space, python's built-in `" ".join(text.split())` is significantly faster than regex-based `re.sub(r"\s+", " ", text).strip()`. In benchmarks, testing this hot path used in `sanitize_content` showed a ~6.27x speedup (from 1.93s down to 0.31s per 1000 iterations for a long string).
**Action:** Always prefer native string methods over regular expressions for simple whitespace normalization in data pipelines or indexing loops.
