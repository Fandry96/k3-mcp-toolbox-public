1. **Optimize Cosine Similarity in `MatryoshkaIndexer.search` (Stage 2 and Stage 1)**
   - I will modify `antigravity-logicware/src/antigravity/k3_mrl_indexer.py`, `antigravity-logicware/k3_mrl_indexer.py`, and `k3-mcp-toolbox/src/k3_mrl_indexer.py` to use `np.einsum` for computing 1D squared norms instead of fully normalizing the NxD matrix using `np.linalg.norm(..., axis=1)`. This will avoid massive intermediate memory allocations during matrix operations and compute the final score mathematically equivalently using raw dot products scaled by the norms.
   - For `_matrix_short_norm_cache`, I will change it to `_matrix_short_sq_norms_cache` to store 1D squared norms instead of the normalized short matrix.
2. **Run tests**
   - I will run the test suite to ensure the optimization doesn't introduce regressions.
3. **Clean up**
   - I will remove temporary scripts and caches.
4. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**
5. **Create PR**
   - I will use the `submit` tool with the required Bolt PR format.
