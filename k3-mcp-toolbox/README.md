# 🧰 K3 MCP Toolbox

> **"Give your AI Agent the tools it actually needs."**
> A production-ready Model Context Protocol (MCP) Server fleet for Windows.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP Ready](https://img.shields.io/badge/MCP-Ready-green.svg)](https://modelcontextprotocol.io)

## What Is This?

A collection of **5 standalone MCP servers** built with [FastMCP](https://github.com/jlowin/fastmcp) that give coding agents practical superpowers: semantic memory search, process management, document intelligence, local LLM orchestration, and git worktree isolation.

Each server runs as an independent `stdio` process — start only what you need, fail independently, toggle freely.

> **Note:** Path constants in the server files reference the author's local Windows environment. Fork and update paths for your own setup.

---

## 🚀 Server Fleet

| Server | Script | What It Does | Tools |
|:---|:---|:---|:---|
| **k3-mrl-memory** | `servers/k3_mrl_memory.py` | Semantic vector search across 25K+ embedded chunks using Matryoshka (MRL) embeddings | `mrl_search`, `mrl_search_skills`, `mrl_index_stats` |
| **k3-agent-ops** | `servers/k3_agent_ops.py` | Windows process hygiene — zombie reaping, port diagnosis, system health | `ops_kill_zombies`, `ops_check_ports`, `ops_free_port`, `ops_system_health` |
| **k3-doc-intel** | `servers/k3_doc_intel.py` | PDF/DOCX text extraction, paragraph-boundary chunking, Gemini embeddings, project-scoped search | `doc_extract_text`, `doc_chunk_and_embed`, `doc_search`, `doc_list_ingested` |
| **k3-local-llm** | `servers/k3_local_llm.py` | Local `llama-server.exe` lifecycle management, GGUF model registry, local embeddings & completions | `local_models_list`, `local_server_status`, `local_server_start`, `local_embed`, `local_complete` |
| **k3-worktree-ops** | `servers/k3_worktree_ops.py` | Git worktree creation, diff inspection, merge-and-cleanup for parallel subagent execution | `worktree_create`, `worktree_list`, `worktree_diff`, `worktree_merge_and_cleanup` |

---

## ⚡ Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Requires: `mcp`, `numpy`, `requests`, `pymupdf`, `python-docx`

### Register a Server

Add to your IDE's MCP config (`mcp_config.json`, `claude_desktop_config.json`, etc.):

```json
{
  "mcpServers": {
    "k3-mrl-memory": {
      "command": "python",
      "args": ["/path/to/k3-mcp-toolbox/servers/k3_mrl_memory.py"]
    }
  }
}
```

Each server is independent — register only the ones you need.

### Environment Variables

| Variable | Required By | Purpose |
|:---|:---|:---|
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `k3-mrl-memory`, `k3-doc-intel` | Gemini Embedding API for query vectors |

---

## 🧪 Testing

Run the full integration suite (24 tests across all 5 servers):

```bash
python test_mcp_fleet.py
```

---

## 🏗️ Architecture

```
k3-mcp-toolbox/
├── servers/                    # 5 standalone FastMCP servers
│   ├── k3_mrl_memory.py       # Semantic vector search (MRL 768-dim)
│   ├── k3_agent_ops.py        # Windows process sentinel
│   ├── k3_doc_intel.py        # Document extraction & embedding
│   ├── k3_local_llm.py        # llama.cpp server orchestrator
│   └── k3_worktree_ops.py     # Git worktree isolation
├── src/                        # Legacy MCP server & core libraries
│   ├── mrl_index.pkl           # Pre-built 25K vector index (not in repo)
│   ├── llama_cpp_server/       # Bundled llama-server.exe (not in repo)
│   └── models/                 # GGUF model files (not in repo)
├── test_mcp_fleet.py           # Integration test suite
├── requirements.txt
└── README.md
```

**Design Decisions:**
- **Fleet over monolith**: Each server is its own process for failure isolation, lazy loading, and independent toggling.
- **MRL embeddings**: 768-dim truncated from 3072-dim `gemini-embedding-001` output, L2-normalized. Cosine similarity via `np.dot` on pre-normalized matrix.
- **O(N) top-k**: Uses `np.argpartition` instead of full sort — 10x faster on large indices.

---

## 🤖 Agent Context

This repository includes a `context.md` file optimized for LLM consumption.
If you are using Cursor, Windsurf, or Antigravity, reference it to give your AI full understanding of this codebase.

## 🤝 Contributing

* **No secrets**: Never commit `.env`, API keys, or tokens.
* **Type hints**: Required on all function signatures.
* **Tests**: Must pass `test_mcp_fleet.py` before merge.

*License: MIT*
