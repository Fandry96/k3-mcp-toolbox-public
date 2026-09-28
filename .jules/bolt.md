## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2023-10-18 - [Optimize File Extension Checking with String Match]
**Learning:** Instantiating `pathlib.Path` objects purely to access `.suffix` during large-scale directory traversal (`os.walk`) introduces significant overhead due to object instantiation inside tight loops. In benchmarks, using native string method `file.endswith(tuple(extensions))` is approximately 17x faster than `Path(file).suffix in extensions` for 200k iterations (0.17s vs 3.05s).
**Action:** Always prefer native string operations (e.g., `endswith` with a tuple of extensions) over object instantiation (`Path`) when filtering strings or paths in loops.
