## 2025-02-19 - Matrix Reconstruction Bottleneck in MatryoshkaIndexer
**Learning:** The `MatryoshkaIndexer.search` method was rebuilding the numpy matrix from the dictionary on *every* call. This is an O(N) operation that dominates execution time for frequent searches.
**Action:** Always check if expensive data structures (like numpy matrices derived from dicts) can be cached and only rebuilt on invalidation (write operations).
