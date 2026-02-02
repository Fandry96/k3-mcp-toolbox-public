## 2025-02-21 - [Duplicate Indexer Files]
**Learning:** The codebase contains two identical copies of `k3_mrl_indexer.py` (one in `k3-mcp-toolbox` and one in `antigravity-logicware`). Optimizing one does not automatically optimize the other, and different parts of the system (`server.py`) may rely on the copy you didn't touch.
**Action:** When working on "toolbox" style repos, always `grep` for the filename to ensure you catch all instances of the code, or verify imports to see which version is actually being used in production.
