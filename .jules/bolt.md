## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize File Traversal Suffix Matching]
**Learning:** During large-scale file system traversal (e.g., in `os.walk`), using `pathlib.Path(file).suffix in extensions` causes expensive object instantiation inside tight loops. Benchmarks show that native string matching via `file.endswith(tuple(extensions))` yields an ~30x speedup for this check.
**Action:** When filtering files by extension inside large loops, use native string methods like `endswith()` with a pre-computed tuple of valid extensions rather than instantiating `pathlib.Path` objects.
