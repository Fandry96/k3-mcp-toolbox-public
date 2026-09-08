# K3 MCP Toolbox & Antigravity Logicware

> **"Production Middleware for the Agentic Era."**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Ready](https://img.shields.io/badge/MCP-Ready-green.svg)](https://modelcontextprotocol.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-27%2F27%20passing-brightgreen.svg)]()
[![Gemini 3.8 Ready](https://img.shields.io/badge/Gemini%203.8%20Flash-GA%20Verified-blueviolet.svg)]()

Welcome to the open-source release of the **K3 MCP Toolbox** and **Antigravity Logicware**. These are the production-tested tools and cognitive primitives we use to power autonomous coding agents on Windows and Linux, open-sourced for the agentic AI community.

---

## ⚡ TL;DR

**K3 MCP Toolbox** solves the 5 most frustrating bottlenecks encountered during long-running coding agent sessions:

1. **Blazing-Fast Local Semantic Memory (`k3-mrl-memory`)**: Queries **25,300+ vectors in 0.22 seconds on local CPU** using Matryoshka Representation Learning (MRL), pre-cached normalized matrix dot products, and $O(N)$ candidate selection via `np.argpartition`.
2. **OS & Runtime Sentinel (`k3-agent-ops`)**: Surgically reaps orphaned test browsers (`chromedriver`, `msedgedriver`) and frees locked dev ports (3000, 8080, etc.) without killing your active IDE or agent sessions.
3. **Subagent Git Worktree Isolation (`k3-worktree-ops`)**: Automatically spins up and merges isolated Git worktrees (`.worktrees/<branch>`) with strict binary allowlists and automatic conflict rollbacks, preventing parallel agents from corrupting git staging.
4. **Secure Document Intelligence (`k3-doc-intel`)**: Ingests and chunks PDFs and DOCX files with a custom `_RestrictedUnpickler` whitelist that blocks arbitrary code execution vulnerabilities on untrusted files.
5. **Local LLM Lifecycle (`k3-local-llm`)**: Manages bundled `llama-server.exe` and local GGUF models for offline embeddings and completions with automated child process reaping.
6. **Gemini 3.8 Flash GA Primitives (`antigravity-logicware`)**: Config sanitizers that strip banned legacy sampling parameters (`temperature`, `top_p`, penalties), enforce strict 1:1 `FunctionCall.id` matching, and provide structured `update(previous_step, plan, next_step)` progress declarations.

---

## 🤖 AI Agent Setup Guide (When an AI Pulls This Repo)

If you are an AI assistant or coding agent (Cursor, Claude Desktop, Antigravity, Cline, Windsurf, Aider, or custom ADK loop) asked to set up or use this repository, follow these precise steps:

### 1. Install Dependencies
```bash
# From repository root:
pip install -r k3-mcp-toolbox/requirements.txt
pip install -e antigravity-logicware
```

### 2. Verify Everything (100% Offline Self-Test)
Run both test suites before touching code:
```bash
# Test the 5-server FastMCP fleet (27 assertions):
python k3-mcp-toolbox/test_mcp_fleet.py

# Test Gemini 3.8 Flash primitives (8 assertions):
python antigravity-logicware/test_flash38.py
```
*(All tests run completely offline and require no API keys or running background daemons).*

### 3. Registering Servers into Your MCP Config
Add any or all of the standalone servers from `k3-mcp-toolbox/servers/` into your client configuration (`mcp_config.json` or `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "k3-agent-ops": {
      "command": "python",
      "args": ["<ABSOLUTE_PATH>/k3-mcp-toolbox/servers/k3_agent_ops.py"]
    },
    "k3-mrl-memory": {
      "command": "python",
      "args": ["<ABSOLUTE_PATH>/k3-mcp-toolbox/servers/k3_mrl_memory.py"],
      "env": {
        "GEMINI_API_KEY": "optional_for_live_embedding_generation",
        "MRL_INDEX_PATH": "<ABSOLUTE_PATH>/k3-mcp-toolbox/src/mrl_index.pkl"
      }
    },
    "k3-worktree-ops": {
      "command": "python",
      "args": ["<ABSOLUTE_PATH>/k3-mcp-toolbox/servers/k3_worktree_ops.py"]
    },
    "k3-doc-intel": {
      "command": "python",
      "args": ["<ABSOLUTE_PATH>/k3-mcp-toolbox/servers/k3_doc_intel.py"]
    },
    "k3-local-llm": {
      "command": "python",
      "args": ["<ABSOLUTE_PATH>/k3-mcp-toolbox/servers/k3_local_llm.py"]
    }
  }
}
```

### 4. Critical Operational Rules for AI Agents
- **FastMCP Server Version**: Always ensure `mcp._mcp_server.version = "1.0.0"` is present after instantiating `FastMCP()`. In editable or non-wheel installations, Pydantic 2.x rejects `None` with a `ValidationError` during the client handshake.
- **Gemini 3.8 Flash Calling Contract**: NEVER pass `temperature`, `top_p`, `top_k`, `presence_penalty`, `frequency_penalty`, `candidate_count`, or `thinking_level: "minimal"` to `gemini-3.8-flash`. Behavior is governed exclusively by `thinking_level: "low" | "medium" | "high"`. Use `FlashConfig.sanitize_kwargs()` from `antigravity.flash38_primitives` to strip banned parameters automatically.
- **Echo Tool Call IDs**: Every `FunctionResponse` MUST echo the exact `id` from the originating `FunctionCall`, or the turn is rejected. Use `ToolDispatcher` from `antigravity-logicware` to automate 1:1 ID tracking.

---

## 📦 What's Included?

### 1. `k3-mcp-toolbox/servers/` (The 5 Standalone FastMCP Servers)

| Server | Tools Exposed | Description |
|---|---|---|
| **`k3_agent_ops.py`** | `ops_kill_zombies`, `ops_check_ports`, `ops_free_port`, `ops_system_health` | Process watchdog and port conflict resolution. Surgically clears orphan Selenium/Playwright processes. |
| **`k3_mrl_memory.py`** | `mrl_search`, `mrl_search_skills`, `mrl_index_stats` | Matryoshka Representation Learning vector search across 25,300+ pre-indexed skills, artifacts, and knowledge items. |
| **`k3_worktree_ops.py`** | `worktree_list`, `worktree_create`, `worktree_diff`, `worktree_cleanup`, `worktree_merge_and_cleanup` | Isolated subagent git worktree manager with test verification gates and rollback on conflict. |
| **`k3_doc_intel.py`** | `doc_extract_text`, `doc_chunk_and_embed`, `doc_search`, `doc_list_ingested` | Safe document extraction (PDF/DOCX) with paragraph-bounded chunking and restricted pickle deserialization. |
| **`k3_local_llm.py`** | `local_models_list`, `local_server_status`, `local_server_start`, `local_server_stop`, `local_embed`, `local_complete` | Local `llama-server.exe` daemon orchestration and offline GGUF fallback capabilities. |

### 2. `antigravity-logicware/` (Cognitive Primitives & SDK Helpers)

- **`flash38_primitives.py`**:
  - `FlashConfig`: Validates and strips legacy parameters, mapping to valid thinking levels (`low`, `medium`, `high`).
  - `ToolDispatcher`: Auto-handles the structured `update()` tool, catches handler exceptions so the model can self-correct, and enforces 1:1 `FunctionCall.id` contract matching.
  - `get_update_tool_declaration()`: Formal schema for the structured progress tool replacing free-text chain-of-thought between tool calls.
  - `run_agent_loop()`: Turn-key multi-turn execution loop with retry logic and telemetry.

---

## ⚡ Benchmarks

### Vector Search (MRL CPU Dot-Product)
- **Index size**: 25,313 vectors (768-dim float32)
- **Search time**: **0.22s** on standard desktop CPU
- **Candidate selection**: `np.argpartition` (O(N)) is **10x faster** than naive `np.argsort` (O(N log N)).

### Integration Test Suite
- **Total assertions**: 27/27 passing (`test_mcp_fleet.py`)
- **Execution time**: < 1.8 seconds offline

---

## 🤝 Contributing

We welcome contributions!
- **Type Hints**: Fully typed function signatures are mandatory.
- **Safety**: Never commit `.pkl`, `.gguf`, or API keys. Always use environment variables or Secret Manager.
- **Testing**: Ensure `python test_mcp_fleet.py` passes 27/27 assertions before opening a pull request.

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.

*Maintained by Fandry96 & The Antigravity Team*
