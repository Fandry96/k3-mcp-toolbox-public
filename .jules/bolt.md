## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize File Traversal using native string methods]
**Learning:** During large-scale file system traversal (e.g., in `os.walk`), replacing `pathlib.Path(file).suffix in extensions` with native string matching like `file.endswith(ext_tuple)` avoids expensive object instantiation inside tight loops. Benchmarks show this can yield a ~7x speedup (e.g., from 14.4s down to 1.9s for a large tree).
**Action:** For file traversal logic, use native string matching (`file.endswith(ext_tuple)`) rather than instantiating `pathlib.Path(file).suffix` to avoid object instantiation overhead in loops.
