## 2025-05-23 - [Optimizing Numpy Stack]
**Learning:** Recreating numpy arrays from a list of vectors on every search is a major bottleneck (O(N)).
**Action:** Cache the stacked matrix and invalidate only on index updates.
