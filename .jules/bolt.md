## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize File Traversal Validation in os.walk]
**Learning:** Using `pathlib.Path(file).suffix in extensions` inside large-scale file system traversals (like `os.walk`) creates significant performance overhead due to the instantiation of a `Path` object for every scanned file. Native string matching (`file.endswith(ext_tuple)`) eliminates this overhead, providing a massive speedup (e.g., ~22x faster in benchmarks).
**Action:** When filtering files by extension during deep directory traversal, prioritize native string methods over `pathlib.Path` instantiation inside tight loops.
