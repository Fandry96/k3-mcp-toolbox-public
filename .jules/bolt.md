## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize File Extension Filtering with endswith]
**Learning:** During large-scale file system traversal (e.g., in `os.walk`), using `Path(file).suffix` inside a tight loop causes significant object instantiation overhead. In our benchmarks, using the native string method `file.endswith(extensions_tuple)` was approximately 25x faster than instantiating `pathlib.Path` objects for every file.
**Action:** When filtering files by extension inside large loops, always use `file.endswith(ext_tuple)` with a tuple of extensions rather than parsing the suffix via `pathlib.Path(file).suffix`.
