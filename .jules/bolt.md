## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize File Traversal and Numpy Matrix Norms]
**Learning:**
1. Object instantiation inside tight loops (`Path(file).suffix`) is a major performance bottleneck during large-scale operations like `os.walk`. Native string methods (`file.endswith(ext_tuple)`) provide an ~36x speedup.
2. Calculating 1D squared norms with `np.linalg.norm(..., axis=1)` creates intermediate arrays and is significantly slower than using `np.sqrt(np.einsum('ij,ij->i', A, A))` which yields a noticeable (~4-5x) speedup. For single 1D vectors, `np.sqrt(np.dot(v, v))` is faster.
**Action:**
1. During file system traversal, always favor native string methods over instantiating expensive objects like `pathlib.Path`.
2. When dealing with large NumPy arrays for distance/similarity calculations, prefer `np.einsum` or `np.dot` over `np.linalg.norm` for 1D squared norms.
