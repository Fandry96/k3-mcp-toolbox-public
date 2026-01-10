# K3 MCP Toolbox: Agent Context

**Role**: You are a Senior Python Engineer maintaining the K3 MCP Toolbox.

## Project Overview

This is a standard Model Context Protocol (MCP) server implementation wrapping three core utilities:

1. **Sequential Thinking**: A metacognition tool for complex reasoning changes (Protocol 310).
2. **MRL Indexer**: A local vector search engine using Matryoshka Representation Learning.
3. **Ops (Zombie Killer)**: A Windows utility to kill stuck processes.

## Architecture

* **Entry Point**: `server.py` (Uses `mcp.server.fastmcp`).
* **Dependencies**: Defined in `requirements.txt`.
* **State**: The MRL indexer persists state to a local `.pkl` file by default.

## Usage Patterns

### Extending Tools

To add a new tool, define a function in `server.py` decorated with `@mcp.tool()`.
Ensure you include complete type hints and docstrings (these become the LLM tool definition).

### MRL Configuration

The indexer is configured in `k3_mrl_indexer.py`.

* Default Model: `models/text-embedding-004`
* Batch Size: `5`
* Save Interval: `20`

## Common Issues

* **Authentication**: The MRL tool requires `GOOGLE_API_KEY` in the environment.
* **Windows execution**: `kill_zombies` uses PowerShell and requires `subprocess` permissions.
