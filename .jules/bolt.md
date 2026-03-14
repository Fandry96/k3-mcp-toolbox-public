## 2025-02-24 - [Optimize string whitespace normalization]
**Learning:** Replacing regex-based whitespace normalization (`re.sub(r"\s+", " ", text).strip()`) with string split/join (`" ".join(text.split())`) provides a significant (~5x-6.7x) speedup in Python. This is especially useful in text processing hot paths, like the `sanitize_content` function.
**Action:** When normalizing whitespace in Python strings, avoid using regular expressions if standard string methods like `.split()` and `" ".join()` can accomplish the task much faster.

## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.
