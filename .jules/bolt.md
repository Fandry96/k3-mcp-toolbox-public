## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-03-01 - [Optimize File Traversal & Stage 2 Cosine Similarity]
**Learning:**
1) In large-scale file system traversal (`os.walk`), using `Path(file).suffix` within inner loops incurs severe object instantiation overhead. Benchmarks show a ~24x speedup by switching to native string methods (`file.endswith(ext_tuple)`).
2) In Stage 2 MRL High-Res Reranking, allocating an intermediate normalized NxD matrix (`m / np.linalg.norm(m)`) causes unnecessary memory pressure and slowdowns. Computing raw dot products first and scaling by 1D squared norms (`np.einsum('ij,ij->i', m, m)`) yields a ~4x speedup (e.g. 1.3ms down to 0.3ms for 1000 candidates).
**Action:**
1) For high-frequency loop path checking, always prefer native strings (`file.endswith`) over `pathlib.Path`.
2) For high-res reranking cosine similarity calculations, use raw dot products scaled by 1D squared norms (`np.einsum`) instead of allocating intermediate fully normalized matrices.
