from mcp.server.fastmcp import FastMCP
import sys
import os
import subprocess
from pathlib import Path
from typing import Optional

# --- CONFIGURATION & PATHS ---
# Use Pathlib for robust cross-platform path handling
BASE_DIR = Path(__file__).parent.resolve()
SRC_DIR = BASE_DIR / "src"

# Add Logicware to python path (Sibling Directory)
LOGICWARE_DIR = BASE_DIR.parent / "antigravity-logicware" / "src"
sys.path.append(str(SRC_DIR))
sys.path.append(str(LOGICWARE_DIR))

# Initialize Server
mcp = FastMCP("k3-toolbox")

# --- GLOBAL STATE (Lazy Loading) ---
# We store the indexer globally so we only load the heavy .pkl file once
_MRL_INDEXER_INSTANCE = None
_THOUGHT_ENGINE = None

# --- IMPORTS (Resilient) ---
try:
    from antigravity import SequentialThinking, MatryoshkaIndexer

    ST_AVAILABLE = True
    MRL_AVAILABLE = True
except ImportError as e:
    ST_AVAILABLE = False
    MRL_AVAILABLE = False
    sys.stderr.write(f"⚠️ Warning: Logicware Import Failed: {e}\n")


# --- TOOL 1: OPS (Windows) ---
@mcp.tool()
async def kill_zombies_win() -> str:
    """
    Terminates stuck Chrome/Driver processes on Windows.
    Returns the STDOUT/STDERR of the cleanup operation.
    """
    script_path = SRC_DIR / "kill_zombies.ps1"

    if not script_path.exists():
        return f"Error: Ops script missing at {script_path}"

    try:
        # Added 'check=False' to prevent crashing on non-zero exit codes (common in cleanup)
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return f"STATUS: {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    except Exception as e:
        return f"CRITICAL OPS FAILURE: {str(e)}"


# --- TOOL 2: COGNITION (Sequential Thinking) ---
@mcp.tool()
def sequential_thinking(
    thought: str,
    next_thought_needed: bool,
    thought_number: int,
    total_thoughts: int,
    is_revision: bool = False,
    revises_thought: Optional[int] = None,
    branch_from_thought: Optional[int] = None,
    branch_id: Optional[str] = None,
    needs_more_thoughts: bool = False,
) -> str:
    """
    Protocol 310: Dynamic problem-solving. Use this to structure complex reasoning.
    Maintains a persistent thought history for the session.
    """
    global _THOUGHT_ENGINE

    if not ST_AVAILABLE:
        return "Error: Sequential Thinking module unavailable."

    # 1. Initialize Singleton if missing
    if _THOUGHT_ENGINE is None:
        _THOUGHT_ENGINE = SequentialThinking()
        sys.stderr.write("🧠 Cognitive Engine Initialized.\n")

    # 2. Execute Step
    return _THOUGHT_ENGINE.execute(
        thought=thought,
        next_thought_needed=next_thought_needed,
        thought_number=thought_number,
        total_thoughts=total_thoughts,
        is_revision=is_revision,
        revises_thought=revises_thought,
        branch_from_thought=branch_from_thought,
        branch_id=branch_id,
        needs_more_thoughts=needs_more_thoughts,
    )


# --- TOOL 3: MEMORY (MRL Search) ---
@mcp.tool()
def mrl_search(query: str, top_k: int = 5) -> str:
    """
    Semantic Search via Matryoshka Representation Learning (MRL).
    Auto-loads index on first run (Cached).
    """
    global _MRL_INDEXER_INSTANCE

    if not MRL_AVAILABLE:
        return "Error: MRL Indexer module unavailable."

    # 1. Check API Key first
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Error: Missing GOOGLE_API_KEY. Cannot embed query."

    try:
        # 2. Lazy Load & Cache the Indexer
        if _MRL_INDEXER_INSTANCE is None:
            sys.stderr.write("⏳ Loading MRL Index into RAM...\n")
            _MRL_INDEXER_INSTANCE = MatryoshkaIndexer(
                api_key=api_key,
                target_dir=str(BASE_DIR),  # Search relative to this script
                index_file="mrl_index.pkl",
            )

        # 3. Execute Search
        results = _MRL_INDEXER_INSTANCE.search(query, top_k=top_k)

        if not results:
            return "No matching records found."

        # 4. Format for Agent Consumption
        formatted = "\n".join(
            [
                f"--- [Score: {r['score']:.4f}] {Path(r['path']).name} ---\n{r['snippet']}"
                for r in results
            ]
        )
        return formatted

    except Exception as e:
        return f"MRL Search Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
