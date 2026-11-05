## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Avoid Pathlib Instantiation in File Walk Loops]
**Learning:** Using `pathlib.Path(file).suffix` inside tight loops (like large directory walks) incurs significant performance overhead due to object instantiation. In tests, replacing it with the native string method `file.endswith(ext_tuple)` resulted in an ~26x speedup.
**Action:** For simple string operations like file extension checking, particularly inside file system traversal loops, always prefer native Python string methods (`endswith`, `startswith`) over instantiating `pathlib.Path` objects.
