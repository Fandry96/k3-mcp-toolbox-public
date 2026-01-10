import subprocess
import json
import os
import threading
import time
import sys
import logging
import shutil
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

# Configure Logging
import logging.handlers

try:
    from tools.ajt_logger import AJTLogger
except ImportError:
    # Fallback for running script directly from tools/ or without root in path
    from ajt_logger import AJTLogger

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "seer.jsonl")

# Initialize AJT Logger
# New signature uses log_dir, not log_file
# Unified Logger Name -> logs/seer.jsonl
ajt = AJTLogger(logger_name="seer", log_dir=LOG_DIR)
logger = ajt.logger

# Console Handler (keep for debugging)
c_handler = logging.StreamHandler()
c_handler.setFormatter(logging.Formatter("[MCP] %(message)s"))
logger.addHandler(c_handler)


@dataclass
class MCPResponse:
    # ------------------------------------------------------------------------------
    # K3 Firehose: Chrome DevTools MCP Bridge
    # Wraps the official 'chrome-devtools-mcp' protocol.
    # ------------------------------------------------------------------------------
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class MCPBridge:
    """
    Robust Stdio Client for Model Context Protocol (MCP) Servers.
    Updated to support 'chrome-devtools-mcp'.
    """

    def __init__(
        self,
        cwd: Optional[str] = None,
        server_package: str = "chrome-devtools-mcp",
        command_override: Optional[List[str]] = None,
    ):
        self.cwd = cwd or os.getcwd()
        self.server_package = server_package
        self.process: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()

        # Dispatcher State
        self._pending_requests: Dict[int, Dict[str, Any]] = {}
        self._shutdown_event = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

        # Bootstrap
        self.executable_cmd = (
            command_override if command_override else self._resolve_executable()
        )
        self._start_server()

    def _resolve_executable(self) -> List[str]:
        """
        Locates the MCP server executable.
        Prioritizes local node_modules, then npx fallback.
        """
        # 1. Check local node_modules (.bin)
        # The binary name often matches the package name or has 'mcp-' prefix
        possible_names = [
            self.server_package,
            f"mcp-{self.server_package.replace('mcp-', '')}",
            "chrome-devtools-mcp",
        ]

        for name in possible_names:
            local_bin = os.path.join(self.cwd, "node_modules", ".bin", name)
            if sys.platform == "win32":
                local_bin += ".cmd"

            if os.path.exists(local_bin):
                logger.info(f"Found local binary: {local_bin}")
                return [local_bin]

        # 2. Check for direct JS file in node_modules (if binary missing)
        # Common pattern: node_modules/package_name/dist/index.js
        js_path = os.path.join(
            self.cwd, "node_modules", self.server_package, "dist", "index.js"
        )
        if os.path.exists(js_path):
            logger.info(f"Found direct JS entry: {js_path}")
            return ["node", js_path]

        # 3. Fallback: Use npx (slower startup, but reliable)
        # We use -y to avoid prompts
        logger.warning(
            f"Local binary not found. Falling back to npx for {self.server_package}"
        )
        return ["npx", "-y", self.server_package]

    def _start_server(self):
        """Spawns the Node.js MCP Server."""
        logger.info(f"Igniting MCP Link: {self.executable_cmd}")
        try:
            self.process = subprocess.Popen(
                self.executable_cmd,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                bufsize=0,  # Unbuffered
            )

            # Start the Dispatcher
            self._reader_thread = threading.Thread(
                target=self._io_loop, daemon=True, name="MCP-Dispatcher"
            )
            self._reader_thread.start()

            # Start Stderr Monitor (Logging)
            threading.Thread(
                target=self._monitor_stderr, daemon=True, name="MCP-Logger"
            ).start()

        except Exception as e:
            logger.critical(f"Failed to start MCP Server: {e}")
            raise

    def _io_loop(self):
        """Reads stdout, parses JSON-RPC, and routes payload."""
        while not self._shutdown_event.is_set() and self.process:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break  # EOF

                line = line.strip()
                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    # Some servers print non-JSON logs to stdout occasionally
                    logger.debug(f"Raw Output: {line[:100]}")
                    continue

                # Routing Logic
                if "id" in payload and payload["id"] in self._pending_requests:
                    req_id = payload["id"]
                    req_ctx = self._pending_requests[req_id]

                    if "error" in payload:
                        req_ctx["response"] = MCPResponse(error=payload["error"])
                    else:
                        req_ctx["response"] = MCPResponse(result=payload.get("result"))

                    req_ctx["event"].set()

                elif "method" in payload:
                    logger.debug(f"Notification: {payload['method']}")

            except Exception as e:
                logger.error(f"Dispatcher Error: {e}")

    def _monitor_stderr(self):
        while self.process and self.process.poll() is None:
            line = self.process.stderr.readline()
            if line:
                logger.info(f"[SERVER] {line.strip()}")

    def send_request(
        self, method: str, params: Dict[str, Any] = {}, timeout: float = 30.0
    ) -> Dict[str, Any]:
        if not self.process:
            raise RuntimeError("MCP Server is down.")

        req_id = int(time.time() * 1000000)
        event = threading.Event()

        with self.lock:
            self._pending_requests[req_id] = {"event": event, "response": None}
            payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            }
            try:
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                raise RuntimeError("MCP Server pipe broken.")

        if not event.wait(timeout):
            with self.lock:
                self._pending_requests.pop(req_id, None)
            raise TimeoutError(f"MCP Request '{method}' timed out")

        ctx = self._pending_requests.pop(req_id)
        response: MCPResponse = ctx["response"]

        if response.error:
            raise Exception(
                f"MCP Error {response.error.get('code')}: {response.error.get('message')}"
            )

        return response.result

    def list_tools(self) -> List[Dict[str, Any]]:
        resp = self.send_request("tools/list")
        return resp.get("tools", [])

    def is_alive(self) -> bool:
        """Checks if the underlying process is still running."""
        return self.process is not None and self.process.poll() is None

    def call_tool(
        self, name: str, arguments: Dict[str, Any], timeout: float = 30.0
    ) -> str:
        resp = self.send_request(
            "tools/call", {"name": name, "arguments": arguments}, timeout=timeout
        )
        content = resp.get("content", [])
        text_parts = [c["text"] for c in content if c["type"] == "text"]
        return "\n".join(text_parts)

    def teardown(self):
        self._shutdown_event.set()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


# --- SMOLAGENTS INTEGRATION ---

try:
    from smolagents import tool
except ImportError:

    def tool(f):
        return f


_bridges: Dict[str, MCPBridge] = {}


def get_bridge(server_name: str = "chrome-devtools-mcp") -> MCPBridge:
    global _bridges

    # 1. Check if we have an existing bridge
    if server_name in _bridges:
        bridge = _bridges[server_name]
        # 2. Verify it's still alive
        if bridge.is_alive():
            return bridge
        else:
            logger.warning(f"Bridge '{server_name}' is dead. Restarting...")
            # Optional: explicit teardown to be safe
            try:
                bridge.teardown()
            except Exception:
                pass
            del _bridges[server_name]

    # 3. Create new bridge if missing or dead
    if server_name not in _bridges:
        cwd = os.getcwd()
        if server_name == "notebooklm":
            # Point to local notebooklm-mcp-main
            cwd = os.path.join(cwd, "notebooklm-mcp-main")
            # For local repo, we run the built JS directly
            dist_path = os.path.join(cwd, "dist", "index.js")
            bridge = MCPBridge(
                cwd=cwd,
                server_package="notebooklm-mcp",
                command_override=["node", dist_path],
            )
        else:
            # Default to devtools
            bridge = MCPBridge(server_package="chrome-devtools-mcp")

        _bridges[server_name] = bridge

    return _bridges[server_name]


@tool
def get_chrome_console_logs(dummy: str = "ignore") -> str:
    """
    Retrieves the browser console logs from the active Chrome tab.
    Args:
        dummy: Ignored argument.
    """
    try:
        bridge = get_bridge()
        # Mapped to 'list_console_messages' from chrome-devtools-mcp
        logs = bridge.call_tool("list_console_messages", {"pageSize": 100})

        # Parse the text output or just return it
        return logs

    except Exception as e:
        return f"Failed to retrieve logs: {str(e)}"


@tool
def navigate_browser(url: str) -> str:
    """
    Navigates the active Chrome tab to a specific URL.
    Args:
        url: The target URL (must include http/https).
    """
    try:
        bridge = get_bridge()

        # AJT: Log Judgment BEFORE execution
        ajt.log_judgment(
            model="seer-automata",
            decision="allow",
            risk_level="low",
            reason=f"Navigating to {url}",
            metadata={"url": url, "tool": "navigate_browser"},
        )

        # Mapped to 'navigate_page' from chrome-devtools-mcp
        # Requires 'type'="url" and the 'url' parameter
        return bridge.call_tool("navigate_page", {"type": "url", "url": url})
    except Exception as e:
        return f"Navigation failed: {str(e)}"


if __name__ == "__main__":
    print("Initializing Bridge...")
    try:
        b = get_bridge("chrome-devtools-mcp")
        print("Bridge Online.")
        print("Discovering Tools...")
        tools = b.list_tools()
        print(f"Found {len(tools)} tools.")

        print("Testing Log Retrieval...")
        # Since we are in the main block, we can call the tool wrapper directly if we want,
        # but better to test the bridge directly or the wrapper.
        # Let's test the wrapper function logic manually via bridge for clarity in smoke test
        logs = b.call_tool("list_console_messages", {"pageSize": 5})
        print(f"Logs (First 50 chars): {str(logs)[:50]}...")

        # Test AJT Logging via Navigation
        print("Testing AJT Logging (Navigation)...")
        # We need to call the tool function wrapper, NOT the bridge directly,
        # because the wrapper has the ajt.log_judgment() call!
        # But wait, the wrapper `navigate_browser` is defined in the global scope.
        nav_result = navigate_browser(url="https://example.com")
        print(f"Navigation Result: {nav_result}")

        print("✅ Smoke Test Passed.")
    except Exception as e:
        print(f"Test Failed: {e}")
    finally:
        for b in _bridges.values():
            b.teardown()
