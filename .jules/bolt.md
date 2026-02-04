## 2024-05-22 - Python Package Name Conflict
**Learning:** The project uses a package named `antigravity`, which conflicts with a standard library module. This caused import errors during benchmarking until `sys.path` was manipulated and `sys.modules` cleaned.
**Action:** When working with Python projects, check for package name conflicts with stdlib early. If found, ensure `sys.path` prioritization is correct or use relative imports carefully.

## 2024-05-22 - Matrix Re-computation Overhead
**Learning:** `MatryoshkaIndexer` was re-stacking numpy arrays from a dictionary on every search. For 50k vectors, this added ~0.1s overhead per search (37% of total time).
**Action:** Always cache derived data structures (like matrices) if the underlying data (index) changes infrequently compared to read operations.
