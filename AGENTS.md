# AGENTS.md — k3-mcp-toolbox-public

## Project Overview

This is the K3 MCP Toolbox — a collection of MCP (Model Context Protocol) servers and tools for AI agent infrastructure. The core component is the **MatryoshkaIndexer** (MRL Indexer), a Matryoshka Representation Learning-based vector search engine.

## Already Merged Optimizations (DO NOT RE-ATTEMPT)

The following optimizations are **already merged into master**:

- **PR #22**: Matrix caching in `MatryoshkaIndexer.search` — caches `np.stack` and normalized matrix, invalidated on index modification. ~86% faster search.
- **PR #27**: Replaced `np.argsort` with `np.argpartition` for top-K selection — O(N) instead of O(N log N). ~90% faster candidate selection.
- **SAVE_INTERVAL** increased to reduce I/O overhead.

## Architecture

- `k3-mcp-toolbox/src/k3_mrl_indexer.py` — Primary MRL indexer implementation
- `antigravity-logicware/k3_mrl_indexer.py` — Mirror implementation (keep in sync)
- Both files should always have matching optimizations.

## Deployment Context

- This repo is cloned into `K3_Firehose` at `c:\K3_Firehose\k3-mcp-toolbox-public\`
- Used as an embedded module by K3 agents for semantic vector search
- Secrets must NEVER be hardcoded — use environment variables or GCP Cloud Secret Manager

## Priority Next Tasks

1. Build an MCP server wrapper around MatryoshkaIndexer
2. Create auto-indexing daemon for file watching
3. Add comprehensive unit tests for search correctness
4. Add type hints and docstrings throughout

## Rules

- All secrets via environment variables or GCP Cloud Secret Manager — NEVER hardcode
- Keep both indexer files in sync
- Include benchmarks in PR descriptions for performance changes
- Check existing merged PRs before starting optimization work to avoid duplicates
