# AGENTS.md — k3-mcp-toolbox-public

## Project Overview

This is the K3 MCP Toolbox — a collection of MCP (Model Context Protocol) servers and tools for AI agent infrastructure. The core component is the **MatryoshkaIndexer** (MRL Indexer), a Matryoshka Representation Learning-based vector search engine.

## Already Merged Optimizations (DO NOT RE-ATTEMPT)

The following optimizations are **already merged into master**:

- **PR #22**: Matrix caching in `MatryoshkaIndexer.search` — caches `np.stack` and normalized matrix, invalidated on index modification. ~86% faster search.
- **PR #27**: Replaced `np.argsort` with `np.argpartition` for top-K selection — O(N) instead of O(N log N). ~90% faster candidate selection.
- **SAVE_INTERVAL** increased to reduce I/O overhead.

## Architecture

- `k3-mcp-toolbox/servers/` — 5 production FastMCP servers:
  - `k3_mrl_memory.py`: 25,000+ vector MRL search with category filters & skill discovery
  - `k3_agent_ops.py`: Windows runtime sentinel, zombie reaping & port diagnosis
  - `k3_doc_intel.py`: Document extraction (PDF/DOCX), boundary chunking & embeddings
  - `k3_local_llm.py`: Bundled `llama-server.exe` lifecycle & local GGUF models
  - `k3_worktree_ops.py`: Git worktree isolation protocol for concurrent agents
- `k3-mcp-toolbox/src/k3_mrl_indexer.py` — Primary MRL indexer implementation
- `antigravity-logicware/src/antigravity/flash38_primitives.py` — Gemini 3.8 Flash agentic primitives (FlashConfig, ToolDispatcher, update tool, ID contract)
- `k3-mcp-toolbox/test_mcp_fleet.py` — 25-assertion integration test suite for the server fleet

## Deployment Context

- This repo is cloned into `K3_Firehose` at `c:\K3_Firehose\k3-mcp-toolbox-public\`
- Used as an embedded module by K3 agents for semantic vector search and runtime ops
- Secrets must NEVER be hardcoded — use environment variables or GCP Cloud Secret Manager

## Completed Milestones (September 2026)

1. [x] FastMCP server wrapper around MatryoshkaIndexer (`servers/k3_mrl_memory.py`)
2. [x] 5-server FastMCP fleet with independent process execution
3. [x] Comprehensive integration test suite (`test_mcp_fleet.py` 27/27 passing)
4. [x] Gemini 3.8 Flash agentic primitives library (`antigravity-logicware`)
5. [x] FastMCP Pydantic 2.x `server_version` fix across all servers
6. [x] `llms.txt` and AI agent onboarding setup guide

## Priority Next Tasks

1. Create auto-indexing daemon / background watcher for live incremental MRL index updates
2. Add generative GGUF model download helper for `k3-local-llm` offline completions
3. Benchmark local llama-server embedding throughput vs Gemini REST API
4. Add type hints and docstrings throughout legacy modules

## Rules

- All secrets via environment variables or GCP Cloud Secret Manager — NEVER hardcode
- Keep both indexer files in sync
- Include benchmarks in PR descriptions for performance changes
- Check existing merged PRs before starting optimization work to avoid duplicates
