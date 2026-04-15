## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-04-15 - [Optimize Whitespace Normalization]
**Learning:** Using regex `re.sub(r"\s+", " ", text).strip()` for standardizing whitespace in large blocks of text is unnecessarily slow compared to Python's built-in string methods. Replacing it with `" ".join(text.split())` achieves the exact same normalization outcome while yielding a ~3.6x speedup in benchmarking (4.31s down to 1.18s for 1000 iterations on large text).
**Action:** Default to `" ".join(text.split())` instead of regex when sanitizing standard whitespace across large text buffers.
