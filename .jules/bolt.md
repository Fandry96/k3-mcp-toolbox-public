## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-10-27 - [Optimize File Traversal with Native String Methods]
**Learning:** During large-scale file system traversal (e.g., in `os.walk`), using object instantiation like `pathlib.Path(file).suffix in extensions` inside the inner loop is extremely slow due to object creation overhead. Benchmarks show that replacing it with native string matching like `file.endswith(tuple(extensions))` yields an ~35x speedup.
**Action:** Avoid instantiating objects inside tight inner loops. Prefer native string methods or built-in primitive operations when processing thousands of iterations.
