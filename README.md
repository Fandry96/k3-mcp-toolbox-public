# K3 MCP Toolbox & Logicware

> **"Middleware for the Agentic Era."**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Ready](https://img.shields.io/badge/MCP-Ready-green.svg)](https://modelcontextprotocol.io)
[![Status: Beta](https://img.shields.io/badge/Status-Beta-orange.svg)]()

Welcome to the open-source release of the **K3 MCP Toolbox** and **Antigravity Logicware**. These are the core tools we use internally to power our Agentic IDE on Windows, extracted into a standalone package for the community.

## 📦 What's Included?

This repository houses two distinct packages:

### 1. [K3 MCP Toolbox](./k3-mcp-toolbox)

**A Windows-native MCP Server implementation.**

- **FastMCP Server**: Lightweight, async server base.
- **Process Management**: `kill_zombies` for cleaning stuck processes (Selenium/Playwright).
- **System Bridge**: Clipboard access, local file ops, and more.

### 2. [Antigravity Logicware](./antigravity-logicware)

**Cognitive protocols for smarter agents.**

- **Sequential Thinking**: Python implementation of Protocol 310 (Chain of Thought).
- **MRL Indexer**: Matryoshka Representation Learning for efficient local search.
- **Three-Brain Tournament**: Framework for adversarial decision making.

### 3. [Docker MCP Gateway](./docker-examples)

**Run MCP Servers anywhere.**

- **Dynamic Loading**: Hot-load tools without restarting the host.
- **Isolation**: Run tools in clean, reproducible containers.

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/Fandry96/k3-mcp-toolbox-public.git
cd k3-mcp-toolbox-public

# Install dependencies (choose your flavor)
pip install -r k3-mcp-toolbox/requirements.txt
```

### Configuration (Claude Desktop)

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "k3-toolbox": {
      "command": "python",
      "args": ["/absolute/path/to/k3-mcp-toolbox-public/k3-mcp-toolbox/server.py"]
    }
  }
}
```

---

## 🐳 Docker Usage

To run the MCP Gateway with Docker:

```bash
cd docker-examples
docker-compose up -d
```

See [docker-examples/README.md](./docker-examples/README.md) not created yet? wait. Just check the folder structure.

---

## 🤝 Contributing

We welcome contributions! Please see the `CONTRIBUTING.md` file in each sub-package for specific guidelines.

- **Hygiene**: Type hints are mandatory.
- **Testing**: Ensure all tests pass before submitting a PR.

---

*Maintained by Fandry96 & The Antigravity Team*
