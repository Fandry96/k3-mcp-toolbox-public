# 🧰 K3 MCP Toolbox

> **"Give your AI Agent the tools it actually needs."**
> A production-ready Model Context Protocol (MCP) Server for Windows & Docker.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP Ready](https://img.shields.io/badge/MCP-Ready-green.svg)](https://modelcontextprotocol.io)

## ⚡ Quick Start (The "30-Second" Rule)

**Installation**

```bash
pip install -r requirements.txt
```

**Run Server**

```bash
# Exposes the server on stdio (works with Claude Desktop / Cursor)
python server.py
```

**Configure (Claude Desktop / Cursor)**
Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "k3-toolbox": {
      "command": "python",
      "args": ["/absolute/path/to/k3-mcp-toolbox/server.py"]
    }
  }
}
```

---

## 🚀 K3 Custom MCP Server Fleet (`servers/`)

In addition to the legacy `k3-toolbox`, the repository now features **5 standalone, production-hardened MCP servers** registered in `mcp_config.json`:

| Server Name | Entry Script | Primary Capabilities | Key Tools |
|:---|:---|:---|:---|
| **`k3-mrl-memory`** | `servers/k3_mrl_memory.py` | High-speed semantic vector search across 25K+ K3 chunks | `mrl_search`, `mrl_search_skills`, `mrl_index_stats` |
| **`k3-agent-ops`** | `servers/k3_agent_ops.py` | Windows runtime hygiene, zombie reaping & port management | `ops_kill_zombies`, `ops_check_ports`, `ops_free_port`, `ops_system_health` |
| **`k3-doc-intel`** | `servers/k3_doc_intel.py` | PDF/DOCX text extraction, chunking & project vector search | `doc_extract_text`, `doc_chunk_and_embed`, `doc_search`, `doc_list_ingested` |
| **`k3-local-llm`** | `servers/k3_local_llm.py` | Bundled `llama-server.exe` orchestration & local embeddings | `local_models_list`, `local_server_status`, `local_server_start`, `local_embed`, `local_complete` |
| **`k3-worktree-ops`** | `servers/k3_worktree_ops.py` | Git worktree isolation protocol for concurrent subagents | `worktree_create`, `worktree_list`, `worktree_diff`, `worktree_merge_and_cleanup` |

### Testing the Fleet
Run the comprehensive integration test suite:
```bash
python test_mcp_fleet.py
```

---

## 🤖 Agent Context (AI-Ready)

This repository includes a `context.md` file optimized for LLM consumption.
If you are using Cursor or Windsurf, simply `@context.md` to give your AI full understanding of this codebase.

## 🤝 Contributing

We follow the **K3-9000 Growth Protocol**.

* **Hygiene**: No `.env` commits.
* **Style**: Type hints required.
* **Tests**: Must pass `test_mcp_health.py`.

*License: MIT*
