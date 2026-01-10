"""
Dynamic MCP Client - Runtime Tool Discovery for K3 Agents

Uses subprocess to communicate with MCP servers via stdio protocol,
matching how the Docker MCP Gateway works internally.

Architecture:
  Agent (Opal) -> MCPServerClient -> MCP Server (stdio) -> Tool Execution

Usage:
  client = MCPServerClient("fetch")
  result = client.call_tool("fetch", {"url": "https://example.com"})
"""

import subprocess
import json
import logging
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


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

    Spawns the server process and communicates via JSON-RPC over stdin/stdout.
    """

    def __init__(self, server_name: str, command: List[str] = None):
        """
        Args:
            server_name: Name of the MCP server
            command: Command to spawn the server. If None, uses docker mcp.
        """
        self.server_name = server_name
        self.command = command
        self._process: Optional[subprocess.Popen] = None
        self._tools_cache: Optional[List[MCPTool]] = None
        self._request_id = 0

    def _get_command(self) -> List[str]:
        """Get the command to spawn the MCP server."""
        if self.command:
            return self.command
        # Use docker to run the server container
        return ["docker", "exec", "-i", f"mcp-{self.server_name}", "mcp-server"]

    def _start_process(self):
        """Start the MCP server process."""
        if self._process is None or self._process.poll() is not None:
            cmd = self._get_command()
            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                logger.debug(f"Started MCP server: {self.server_name}")
            except Exception as e:
                logger.error(f"Failed to start MCP server {self.server_name}: {e}")
                raise

    def _send_request(
        self, method: str, params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        self._start_process()

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }

        try:
            # Send request
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()

            # Read response
            response_line = self._process.stdout.readline()
            if not response_line:
                return {"error": "No response from server"}

            response = json.loads(response_line)

            if "error" in response:
                return {"error": response["error"]}

            return response.get("result", {})

        except Exception as e:
            logger.error(f"MCP request failed: {e}")
            return {"error": str(e)}

    def list_tools(self) -> List[MCPTool]:
        """Discover available tools from this MCP server."""
        if self._tools_cache:
            return self._tools_cache

        result = self._send_request("tools/list")

        tools = []
        for tool_data in result.get("tools", []):
            tool = MCPTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                server=self.server_name,
                input_schema=tool_data.get("inputSchema", {}),
            )
            tools.append(tool)

        self._tools_cache = tools
        return tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool on this MCP server."""
        return self._send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )

    def close(self):
        """Close the MCP server process."""
        if self._process:
            self._process.terminate()
            self._process = None


class MCPToolRegistry:
    """
    Registry for multiple MCP servers.

    Aggregates tools from all registered servers for unified access.
    """

    # Default servers from Docker MCP config
    DEFAULT_SERVERS = {
        "fetch": {
            "command": ["npx", "-y", "@anthropic/mcp-fetch"],
            "description": "HTTP fetch operations",
        },
        "playwright": {
            "command": ["npx", "-y", "@anthropic/mcp-playwright"],
            "description": "Browser automation with Playwright",
        },
    }

    def __init__(self):
        self._clients: Dict[str, MCPServerClient] = {}
        self._all_tools: List[MCPTool] = []

    def register_server(
        self,
        name: str,
        command: List[str] = None,
    ) -> "MCPToolRegistry":
        """Register an MCP server."""
        self._clients[name] = MCPServerClient(name, command)
        return self

    def register_default_servers(self) -> "MCPToolRegistry":
        """Register the default MCP servers."""
        for name, config in self.DEFAULT_SERVERS.items():
            self.register_server(name, config.get("command"))
        return self

    def discover_all_tools(self) -> List[MCPTool]:
        """Discover tools from all registered servers."""
        self._all_tools = []
        for name, client in self._clients.items():
            try:
                tools = client.list_tools()
                self._all_tools.extend(tools)
                logger.info(f"Discovered {len(tools)} tools from {name}")
            except Exception as e:
                logger.error(f"Failed to discover tools from {name}: {e}")
        return self._all_tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool by name.

        Automatically routes to the correct server based on cached tool info.
        """
        # Find which server has this tool
        for tool in self._all_tools:
            if tool.name == tool_name:
                client = self._clients.get(tool.server)
                if client:
                    return client.call_tool(tool_name, arguments)

        return {"error": f"Tool '{tool_name}' not found in any registered server"}

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        for tool in self._all_tools:
            if tool.name == name:
                return tool
        return None

    def close_all(self):
        """Close all MCP server processes."""
        for client in self._clients.values():
            client.close()


# Convenience function for quick tool invocation
def mcp_fetch(url: str, method: str = "GET") -> str:
    """Quick fetch via MCP fetch server."""
    client = MCPServerClient("fetch", ["npx", "-y", "@anthropic/mcp-fetch"])
    try:
        result = client.call_tool("fetch", {"url": url, "method": method})
        return result.get("content", result)
    finally:
        client.close()


# Smoke test
if __name__ == "__main__":
    print("MCP Tool Registry Smoke Test")
    print("=" * 40)

    # Test with fetch server (npx-based, no docker needed)
    try:
        registry = MCPToolRegistry()
        registry.register_server("fetch", ["npx", "-y", "@anthropic/mcp-fetch"])

        print("\nDiscovering tools...")
        tools = registry.discover_all_tools()
        print(f"Found {len(tools)} tools")

        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")

        registry.close_all()
        print("\n✓ Registry test passed")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
