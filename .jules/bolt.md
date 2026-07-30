## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-07-30 - [Optimize File Traversal with Native String Methods]
**Learning:** During large-scale file system traversal (e.g., in `os.walk`), using `pathlib.Path(file).suffix in extensions` creates unnecessary object instantiation overhead inside tight loops. Replacing this with native string matching like `file.endswith(ext_tuple)` avoids this overhead. Benchmarks show this provides around a 7.6x speedup (e.g. 0.18s down to 0.024s for 10,000 files).
**Action:** Always prefer native string methods for file extension checking inside large loops over instantiating `Path` objects.
