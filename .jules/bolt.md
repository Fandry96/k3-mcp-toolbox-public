## 2024-05-23 - Performance Optimization of MRL Indexer Search

**Learning:** Reconstructing a dense matrix from a list of dictionary values (`np.stack([d['vector'] for d in self.index.values()])`) inside a frequently called search method is a major performance bottleneck, especially as the index grows. For 10,000 vectors, this operation took ~22ms per call. Caching the matrix reduced this to ~6ms (a ~3.6x speedup).

**Action:** When working with vector search or similar data-heavy operations in Python, always look for opportunities to pre-compute or cache the heavy data structures (like numpy matrices) rather than rebuilding them on every query. Ensure cache invalidation logic is robust (invalidate on any write).
