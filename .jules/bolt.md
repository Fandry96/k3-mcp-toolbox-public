## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2024-07-08 - Optimize File Extension Checking in Directory Traversal
**Learning:** Using `Path(file).suffix in extensions` where `extensions` is a set incurs significant overhead from instantiating `Path` objects during large-scale filesystem traversal. Testing showed `Path` instantiation in a hot loop is very slow.
**Action:** Replace `Path(file).suffix in set` with native string matching `file.endswith(tuple)` for checking file extensions inside `os.walk` loops. This simple change yields a ~24x speedup during large directory scans.
