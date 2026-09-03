## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-24 - [Optimize Path Traversal with Native String Methods]
**Learning:** During large-scale file system traversal (e.g., inside `os.walk` loops), creating `pathlib.Path` objects to access the `.suffix` property introduces substantial object instantiation overhead. In benchmarks, switching to a native string method (`file.endswith(ext_tuple)`) resulted in an ~30x speedup for file extension matching.
**Action:** When filtering files by extension inside large traversal loops, always use native string methods like `endswith()` with a tuple of extensions rather than instantiating `pathlib.Path` objects.
