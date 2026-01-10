# Reddit Post Draft

**Subreddits**: r/LocalLLama, r/ArtificialInteligence, r/Python, r/ClaudeAI

## Title Options

1. **[Release] K3-Toolbox: A Production-Grade MCP Server for Sequential Thinking & MRL**
2. **Open Sourcing my "Glass Box" Agentic Stack (Protocol 310 + Gemini MRL)**
3. **Bringing "System 2" Thinking to Claude Desktop via MCP (Python/FastMCP)**

---

## Body Content

**TL;DR:** I’ve open-sourced the cognitive engine I use for my autonomous agents. It’s a standalone MCP Server (Python) that adds **Sequential Thinking** (deep reasoning) and **Matryoshka Representation Learning** (efficient vector memory) to any MCP-compliant client (Claude Desktop, Cursor, etc.).

### The Problem

Most agents (including Claude) default to linear, "System 1" thinking. They hallucinate when complexity increases because they don't have a rigid structure to check their work or a persistent memory state to track multi-step logic.

### The Solution: K3-Toolbox

I built a unified MCP Server (`server.py`) that exposes three critical tools:

1. **SequentialThinking (Protocol 310)**: A structured cognitive process that forces the agent to break problems down, revise its own thoughts, and branch its reasoning *before* committing to an answer. It maintains state across the session ("Glass Box" auditing).
2. **MRL Search (The Librarian)**: A local vector search engine using Google's **Text-Embedding-004**. It uses Matryoshka Representation Learning (MRL) for high-efficiency reranking (64-dim funnel -> 768-dim ranking).
3. **Ops Utilities**: Scripts to handle local process management (e.g., terminating stuck web drivers) without crashing the host.

### Architecture

* **Zero-Friction**: Logic is bundled into `src/`—no complex package installs needed.
* **Performance**: Lazy-loading singletons for the vector index (0ms startup time until used).
* **Compatibility**: Works natively with Claude Desktop via `stdio`.

### How to Use

1. Clone the repo: `git clone https://github.com/[YOUR_USERNAME]/k3-mcp-toolbox`
2. Install `mcp[cli] google-genai numpy`
3. Add to your Claude Config:

```json
"k3-toolbox": {
  "command": "python",
  "args": ["/path/to/k3-mcp-toolbox/server.py"],
  "env": { "GOOGLE_API_KEY": "..." }
}
```

### Why Logicware?

As we move toward autonomous agents, **observability** is key. This toolbox is designed to be a "Glass Box"—you see exactly *how* the agent reached its conclusion, not just the final output.

**Link**: [GitHub Repository URL]

---
*Questions and feedback on the implementation (specifically the MRL chunking) are welcome.*
