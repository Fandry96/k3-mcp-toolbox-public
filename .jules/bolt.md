## 2025-05-15 - Matrix Reconstruction Overhead
**Learning:** Reconstructing numpy arrays from a list of vectors on every search call in `MatryoshkaIndexer` caused significant overhead (12ms vs 0.7ms).
**Action:** Always look for expensive operations inside frequently called methods (like `search`) that can be cached if the underlying data doesn't change often.
