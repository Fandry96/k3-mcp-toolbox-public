# 🧠 Antigravity Logicware

> **"Middleware for Agentic Thought."**
> A collection of high-assurance cognitive protocols for Python AI Agents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Protocol 310](https://img.shields.io/badge/Protocol-310-purple.svg)](https://modelcontextprotocol.io)

## ⚡ Quick Start (The "30-Second" Rule)

**Installation**

```bash
# Just copy the files or install via pip (coming soon)
pip install numpy google-genai
```

**Usage: Sequential Thinking**

```python
from sequential_thinking import SequentialThinking

engine = SequentialThinking()
print(engine.execute(
    thought="Analyzing the user request...",
    thought_number=1,
    total_thoughts=3,
    next_thought_needed=True
))
```

---

## 🧩 The Protocols

### 1. Sequential Thinking (`sequential_thinking.py`)

*Based on Protocol 310.*
Enables "System 2" metacognition. Agents can:

* **Branch**: Explore alternative hypotheses.
* **Revise**: Correct previous mistakes explicitly.
* **Structure**: Force a multi-step planning phase before execution.

### 2. Matryoshka Search (`k3_mrl_indexer.py`)

*Based on MRL Research (Kusupati et al).*
A vector search engine that fits in a single Python file.

* **Funnel Search**: Uses 64-dim embeddings for fast filtering, 768-dim for precision.
* **Gemini-Native**: Optimized for `text-embedding-004`.

### 3. Three-Brain Tournament (`tournament_logic.py`)

A governance framework for high-stakes decisions.

* **Proposer**: Generates ideas.
* **Critic**: Attacks ideas.
* **Judge**: Decides the winner.

---

## 🤖 Agent Context (AI-Ready)

This repository includes a `context.md` file optimized for LLM consumption.
If you are using Cursor or Windsurf, simply `@context.md` to give your AI full understanding of these cognitive protocols.

## 🤝 Contributing

We follow the **K3-9000 Growth Protocol**.

* **Hygiene**: No `.env` commits.
* **Type Hints**: Required for all logic modules.

*License: MIT*
