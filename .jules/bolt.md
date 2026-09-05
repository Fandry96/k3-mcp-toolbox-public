## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize File Traversal with Native String Matching]
**Learning:** Instantiating `pathlib.Path` objects inside tight loops (like `os.walk`) just to check file extensions (`Path(file).suffix`) creates significant object allocation overhead. In benchmarks, switching to native string methods `file.endswith(tuple)` yielded an ~7x speedup for large directories.
**Action:** During large-scale file system traversal, always prefer native string methods (e.g., `endswith` with a tuple of extensions) over object instantiation for simple suffix checks.
