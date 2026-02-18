## 2026-02-18 - Caching Matrix Construction in MatryoshkaIndexer
**Learning:** Reconstructing a numpy matrix from a list of vectors on every search (`O(N)`) is a significant bottleneck even for moderate index sizes (5000 docs). Caching the matrix reduces search time by ~72% (11ms -> 3ms).
**Action:** Always look for `O(N)` operations inside frequently called methods like `search` that can be cached if the underlying data changes infrequently.
