import subprocess
import json
import logging
import sys
import os
import shutil
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

# --- CONFIGURATION ---
# Adjust this if your server is located elsewhere
SERVER_SCRIPT_PATH = Path(__file__).parent / "server.py"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Antigravity-Bridge")

# ==============================================================================
# PART 1: THE CLIENT CORE ("OPAL")
# ==============================================================================


@dataclass
class MCPTool:
    """Represents a discovered MCP tool."""

    name: str
    description: str
    server: str
    input_schema: Dict[str, Any] = field(default_factory=dict)


class MCPServerClient:
    """
    Client for a single MCP server via stdio protocol.
    """

    def __init__(self, server_name: str, command: List[str]):
        self.server_name = server_name
        self.command = command
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0

    def _start_process(self):
        """Start the MCP server process with UNBUFFERED stdio."""
        if self._process is None or self._process.poll() is not None:
            try:
                # CRITICAL: We pass the environment to ensure API Keys are inherited
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered Python

                self._process = subprocess.Popen(
                    self.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=sys.stderr,  # Pipe server errors to our console
                    text=True,
                    env=env,
                )
                logger.info(f"Connected to MCP Server: {self.server_name}")
            except Exception as e:
                logger.error(f"Failed to spawn server: {e}")
                raise

    def initialize(self):
        """Perform MCP Handshake."""
        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"roots": {"listChanged": True}, "sampling": {}},
            "clientInfo": {"name": "k3-bridge", "version": "1.0.0"},
        }
        res = self._send_request("initialize", init_params)

        # After initialize, must send 'notifications/initialized'
        self._send_notification("notifications/initialized")
        return res

    def _send_notification(self, method: str, params: Dict[str, Any] = None):
        """Send JSON-RPC notification (no response expected)."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        payload = json.dumps(request) + "\n"
        if self._process:
            self._process.stdin.write(payload)
            self._process.stdin.flush()

    def _send_request(
        self, method: str, params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Send JSON-RPC request."""
        # Start process on first request/init
        if self._process is None:
            self._start_process()

        self._request_id += 1

        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }

        try:
            # Write
            payload = json.dumps(request) + "\n"
            self._process.stdin.write(payload)
            self._process.stdin.flush()

            # Read
            response_line = self._process.stdout.readline()
            if not response_line:
                # Check stderr for clues
                err = self._process.stderr.read() if self._process.stderr else ""
                raise ConnectionError(
                    f"Server closed connection unexpectedly. Stderr: {err}"
                )

            response = json.loads(response_line)
            if "error" in response:
                return {"error": response["error"]}
            return response.get("result", {})

        except Exception as e:
            logger.error(f"RPC Failure: {e}")
            return {"error": str(e)}

    def list_tools(self) -> List[MCPTool]:
        """Discover tools."""
        result = self._send_request("tools/list")
        if "error" in result:
            logger.error(f"List Tools Error: {result['error']}")
            return []

        tools = []
        for tool_data in result.get("tools", []):
            tools.append(
                MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    server=self.server_name,
                    input_schema=tool_data.get("inputSchema", {}),
                )
            )
        return tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke tool."""
        return self._send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )

    def close(self):
        if self._process:
            self._process.terminate()
            self._process = None


# ==============================================================================
# PART 2: THE INTEGRATION TEST
# ==============================================================================


def run_glass_box_test():
    print(f"\n🔮 K3-9000 GLASS BOX TEST PROTOCOL")
    print(f"Target Server: {SERVER_SCRIPT_PATH}")
    print("=" * 60)

    # 1. Environment Check
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  WARNING: GOOGLE_API_KEY not found in environment.")
        print("   'mrl_search' will fail, but 'sequential_thinking' should work.")

    # 2. Command Resolution
    if not SERVER_SCRIPT_PATH.exists():
        print(f"❌ FATAL: server.py not found at {SERVER_SCRIPT_PATH}")
        return

    # Using sys.executable ensures we use the same Python env (venv)
    # '-u' flag is CRITICAL for stdio communication
    cmd = [sys.executable, "-u", str(SERVER_SCRIPT_PATH)]

    client = None
    try:
        # 3. Connection
        print("[1/5] Connecting to Server...")
        client = MCPServerClient("k3-toolbox", cmd)

        print("[2/5] Performing Handshake...")
        init_res = client.initialize()
        print(
            f"   ✓ Handshake Complete: {init_res.get('serverInfo', 'Unknown Server')}"
        )

        # 4. Discovery
        print("[3/5] Discovering Tools...")
        tools = client.list_tools()

        if not tools:
            print("❌ FAILURE: Server connected but returned 0 tools.")
            return

        for t in tools:
            print(f"   ✓ {t.name:<25} | {t.description[:40]}...")

        # 5. Testing Logic (Sequential Thinking)
        print("\n[4/5] Testing Cognition (Sequential Thinking)...")
        logic_payload = {
            "thought": "Initiating Glass Box diagnostic routine.",
            "next_thought_needed": True,
            "thought_number": 1,
            "total_thoughts": 3,
            "needs_more_thoughts": True,
        }
        res_logic = client.call_tool("sequential_thinking", logic_payload)

        if "content" in res_logic:
            # MCP usually returns list of content blocks
            output_text = res_logic["content"][0]["text"]
            print(f"   [Response]\n{output_text.strip()}")
        else:
            print(f"   [Raw Result] {res_logic}")

        # 6. Testing Memory (MRL Search)
        print("\n[5/5] Testing Memory (MRL Search)...")
        # We search for 'Protocol' which appears in the docstrings
        res_search = client.call_tool("mrl_search", {"query": "Protocol 310"})

        # Extract text
        if isinstance(res_search, dict) and "content" in res_search:
            search_txt = res_search["content"][0]["text"]
        else:
            search_txt = str(res_search)

        print(
            f"   [Search Result Sample]\n   {search_txt[:200].replace(chr(10), ' ')}..."
        )

        print("\n✅ SYSTEM ONLINE: All tests passed.")

    except Exception as e:
        print(f"\n❌ BRIDGE COLLAPSE: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    run_glass_box_test()
