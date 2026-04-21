## 2025-02-23 - [Optimize NumPy Top-K Selection with argpartition]
**Learning:** `np.argsort` has O(N log N) complexity, causing performance bottlenecks during top-K candidate selection in large vector search indices (e.g., K3 MRL Indexer). For top-K selection without a full sort, `np.argpartition` provides O(N) complexity. In benchmarks, switching from `argsort` to `argpartition` for K=75 out of 100,000 vectors reduced the execution time from ~4.0ms down to ~0.36ms (a 10x+ improvement).
**Action:** When extracting top-K candidates from large NumPy arrays (e.g., scoring matrices, similarity calculations), always prioritize `np.argpartition` followed by sorting just the selected partition, rather than using `np.argsort` on the entire array.

## 2026-04-21 - [Optimize string sanitization using split/join]
**Learning:** Python's native `" ".join(text.split())` is significantly faster (~4.5x) than the regex equivalent `re.sub(r"\s+", " ", text).strip()` for normalizing whitespace, especially over large chunks of text during an indexing hot-path.
**Action:** Default to string split/join for basic whitespace sanitization rather than invoking regex overhead when parsing large texts.
