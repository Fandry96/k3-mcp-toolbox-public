# Antigravity Logicware: Agent Context

**Role**: You are a Cognitive Architect designing AI agent workflows.

## Project Overview

This repository contains "Logicware"—pure Python implementations of advanced reasoning protocols. Unlike standard tools (which do things), these modules help agents *think*.

## Key Modules

### 1. Sequential Thinking (`sequential_thinking.py`)

**Purpose**: To slow down execution and force planning.
**Usage**:
Use `execute()` to log a thought. The output is a formatted string that serves as a "scratchpad" for the LLM's context window.

### 2. MRL Indexer (`k3_mrl_indexer.py`)

**Purpose**: Efficient semantic memory.
**Configuration**:

* `DEFAULT_MODEL`: text-embedding-004
* `BATCH_SIZE`: 5
**Key Methods**:
* `run_indexing(limit=N)`: Scans folder and builds `.pkl` index.
* `search(query, top_k=5)`: Returns ranked snippets.

### 3. Tournament Logic (`tournament_logic.py`)

**Purpose**: Conflict resolution.
**Logic**: Implement a voting or scoring mechanism where a neutral "Judge" persona evaluates outputs from "Proposer" and "Critic" personas.

## Implementation Notes

* **Dependencies**: MRL requires `google-genai`. Sequential Thinking is pure Python (no deps).
* **State**: Thinking history is ephemeral (in-memory). MRL index is persistent (disk).
