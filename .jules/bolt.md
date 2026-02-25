## 2025-05-21 - MatryoshkaIndexer Search Optimization
**Learning:** Repeatedly calling `np.stack` on a large list of arrays (10k+) inside a search loop caused ~85% latency overhead (18ms vs 2.6ms).
**Action:** Implemented lazy matrix caching with invalidation on write. Always look for `np.stack` or `np.array` conversions inside hot loops.
