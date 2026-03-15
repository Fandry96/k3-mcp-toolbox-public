## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2025-02-23 - [Optimize Whitespace Normalization in Python]
**Learning:** Using regex (`re.sub(r"\s+", " ", text).strip()`) for normalizing whitespace is significantly slower than using Python's built-in string splitting and joining (`" ".join(text.split())`). In benchmarks, the string method is ~2.5x to 3.5x faster, which is critical when processing tens of thousands of text chunks during indexing.
**Action:** When normalizing contiguous whitespace and stripping leading/trailing spaces in Python, prefer `" ".join(text.split())` over regex.
