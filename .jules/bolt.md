## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-24 - [Optimize File Extension Checking in OS Walk Loops]
**Learning:** Instantiating `pathlib.Path(file)` inside a tight loop over thousands of files is highly inefficient compared to using native string methods like `file.endswith(ext_tuple)`. In benchmark tests, avoiding object instantiation for extension checking resulted in an ~20x performance improvement (e.g., from 1.44s to 0.07s for 200,000 files).
**Action:** When filtering files during large-scale filesystem traversal, avoid object instantiation in the inner loop. Convert `extensions` sets to tuples and use `str.endswith()` instead of creating `Path` objects.
