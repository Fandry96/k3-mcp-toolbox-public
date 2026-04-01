## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-04-01 - [Optimize Whitespace Normalization with String Split/Join]
**Learning:** The expression `re.sub(r"\s+", " ", text).strip()` is a common but slow way to normalize whitespace in Python strings. A significantly faster approach (measured at ~6x speedup in benchmarking) is using Python's built-in string splitting and joining: `" ".join(text.split())`.
**Action:** When normalizing consecutive whitespaces and stripping leading/trailing whitespace in strings, always prioritize `" ".join(text.split())` over regex substitution to maximize execution speed, especially in hot paths like text sanitization for indexing.
