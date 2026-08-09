## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-24 - [Optimize file traversal with endswith]
**Learning:** During large-scale file system traversal (e.g., in `os.walk`), using `pathlib.Path(file).suffix` inside a tight loop causes significant object instantiation overhead. Replacing it with native string matching (`file.endswith(ext_tuple)`) yields an ~7x to 36x speedup depending on the number of files.
**Action:** For file extension checking in large loops, prioritize native string methods like `str.endswith` with a tuple of extensions over instantiating `Path` objects.
