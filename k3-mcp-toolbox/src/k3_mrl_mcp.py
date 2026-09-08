import os
import sys
import argparse
from pathlib import Path

# Try to load the indexer logic
try:
    from k3_mrl_indexer import MatryoshkaIndexer
except ImportError:
    print(
        "[ERROR] Could not import k3_mrl_indexer. Make sure you run this script from the same directory.",
        file=sys.stderr,
    )
    sys.exit(1)

from mcp.server.fastmcp import FastMCP

# Create the MCP Server
mcp = FastMCP("k3_mrl_memory")
mcp._mcp_server.version = "1.0.0"

# Global reference to the indexer so we only load the 1.2GB model once
INDEXER = None
DEFAULT_WORKSPACE = "c:/K3_Firehose/.agent/skills"
DEFAULT_INDEX = "mrl_index.pkl"


def _ensure_indexer(
    workspace_path: str = DEFAULT_WORKSPACE, index_file: str = DEFAULT_INDEX
):
    global INDEXER
    if INDEXER is None:
        print(
            f"[SYSTEM] Initializing MRL Indexer for MCP Server on path {workspace_path}...",
            file=sys.stderr,
        )
        # api_key is ignored in the sentence_transformers refactor
        INDEXER = MatryoshkaIndexer(
            api_key="none", target_dir=workspace_path, index_file=index_file
        )
    return INDEXER


@mcp.tool()
def search_agent_memory(query: str, top_k: int = 5) -> str:
    """
    Search the agent's memory (skills, blueprints, capabilities) using semantic MRL search.
    Returns the top matching context snippets to help the agent remember how to do things.
    """
    indexer = _ensure_indexer()
    results = indexer.search(query, top_k=top_k)

    if not results:
        return f"No memory found for query: {query}"

    formatted = [f"--- Semantic Memory Search Results for '{query}' ---"]
    for i, res in enumerate(results):
        file_path = res["file_path"]
        score = res["score"]
        snippet = res["snippet"].strip()

        formatted.append(f"\n[Result {i + 1} | Score: {score:.3f}] File: {file_path}")
        formatted.append(snippet + "\n...")

    return "\n".join(formatted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K3 MRL Memory MCP Server")
    parser.add_argument(
        "--path",
        type=str,
        default=DEFAULT_WORKSPACE,
        help="Workspace to serve memory for",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=DEFAULT_INDEX,
        help="Pickle file containing vectors",
    )

    args = parser.parse_args()

    # Pre-warm the indexer before giving control to MCP
    _ensure_indexer(args.path, args.index)

    # Hand off to standard stdio execution
    mcp.run()
