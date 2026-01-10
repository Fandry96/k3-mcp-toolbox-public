# Contributing to Antigravity Logicware

We follow the **K3-9000 Growth Protocol**.

## 1. Hygiene

* **Secrets**: This repository contains pure logic. NEVER commit API keys or `.env` files.
* **Files**: Do not commit `__pycache__` or `*.pkl` (vector indices).

## 2. Code Standards

* **Type Hints**: All protocols must be strictly typed.
* **Docstrings**: Explain the *cognitive purpose* of the class (e.g., "Enables recursive expansion of thought branches").

## 3. Pull Requests

* Keep PRs focused on one protocol at a time.
* If modifying the MRL Indexer, verify backwards compatibility with existing `.pkl` indices.
