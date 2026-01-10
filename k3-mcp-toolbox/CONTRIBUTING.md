# Contributing to K3 MCP Toolbox

We follow the **K3-9000 Growth Protocol**. Please adhere to these standards to ensure your PR is accepted.

## 1. Hygiene (The "Clean Room" Rule)

* **NEVER** commit secrets, API keys, or `.env` files.
* **NEVER** commit `__pycache__`, `.DS_Store`, or `node_modules`.
* We use a strict `.gitignore`. If your PR includes these, it will be closed.

## 2. Documentation

* **Type Hints**: All Python functions must have full type hints.
* **Docstrings**: All `server.py` tools must have docstrings (these are read by the AI).
* **README**: If you add a feature, update the README "Tools" section.

## 3. Architecture

* **Standard Library First**: Try to avoid adding pip dependencies unless absolutely necessary.
* **Windows & Linux**: We verify on both. If adding an OS-specific tool (like `kill_zombies`), wrap it in a try/except block or check `sys.platform`.

## 4. Testing

Run the health check before pushing:

```bash
python test_mcp_health.py
```

If it prints "✅ Smoke Test Passed", you are green.
