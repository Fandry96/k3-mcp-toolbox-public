## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize File Traversal in os.walk]
**Learning:** During large-scale file system traversal (e.g., in `os.walk`), instantiating `pathlib.Path(file).suffix` within a tight loop creates massive object instantiation overhead compared to using native string methods like `file.endswith(ext_tuple)`. In benchmarks, checking 180k strings took ~1.3s with `Path().suffix` and only ~0.037s with `endswith`, representing an ~18x speedup.
**Action:** Replace `Path(file).suffix in extensions` with `file.endswith(ext_tuple)` in `os.walk` operations for significantly faster file filtering.
