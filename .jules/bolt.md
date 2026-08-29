## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Avoid Path Object Instantiation in Tight Loops]
**Learning:** In large-scale file system traversals (e.g., `os.walk`), instantiating `pathlib.Path(file)` for every file just to check its suffix is a major performance bottleneck due to object instantiation overhead.
**Action:** Use native string methods like `file.endswith(ext_tuple)` instead of `Path(file).suffix in extensions` when filtering files in tight loops. Benchmarks show this yields an ~7x to ~36x speedup depending on scale.
