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

## 🛠️ The Tools

### 1. `sequential_thinking` (The Logic)

A "System 2" thinking module. Allows agents to break complex problems into dynamic steps, revise previous thoughts, and branch constraints.

* *Based on Protocol 310.*

### 2. `mrl_search` (The Brain)

**Matryoshka Representation Learning (MRL)** indexer.

* **Zero-Setup**: Uses local storage (no vector DB required).
* **SOTA**: Configured for `text-embedding-004` (Gemini) by default.
* **Fast**: Funnel search implementation (64-dim shortlist -> 768-dim rerank).

### 3. `kill_zombies` (The Ops)

*Windows Only.* Instantly terminates stuck Chrome/Chromedriver processes and releases file locks. Essential for Selenium/Playwright agents.

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
