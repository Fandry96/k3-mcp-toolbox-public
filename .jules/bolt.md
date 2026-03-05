## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-03-05 - [Optimize Whitespace Normalization in Indexing]
**Learning:** In highly repetitive text processing paths (like `sanitize_content` during chunking/indexing), using regex for simple whitespace normalization (`re.sub(r"\s+", " ", text).strip()`) is a significant performance bottleneck.
**Action:** Replace `re.sub` whitespace normalization with Python's native string split and join (`" ".join(text.split())`). In local benchmarks, this yielded a ~6.7x speedup for the operation, significantly reducing the overhead per text chunk during the indexing process.
