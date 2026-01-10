#!/usr/bin/env python3
"""
MCP Health Check - Verify all configured servers are operational
"""

import sys
import os

# Ensure tools directory is in path for imports
sys.path.append(os.getcwd())

try:
    from tools.simple_mcp_client import SimpleMCPClient
except ImportError:
    # Attempt to handle if run from different context, or warn user
    pass


def test_all_servers():
    servers = ["chrome-devtools", "duckduckgo", "filesystem"]
    results = {}

    print("Initializing MCP Health Check...")

    for server in servers:
        try:
            # Note: This assumes SimpleMCPClient supports a mechanism to select servers or
            # uses the default configuration we just recommended.
            # If SimpleMCPClient loads from mcp_config.json, ensure that file is updated first
            # as per the Action Items.
            print(f"Testing connection to: {server}...")
            with SimpleMCPClient(servers=[server], verbose=False) as client:
                tools = client.list_tools()
                results[server] = {"status": "✅ OK", "tools": len(tools)}
        except Exception as e:
            results[server] = {"status": "❌ FAIL", "error": str(e)}

    # Print report
    print("\n=== MCP Health Check ===")
    for server, result in results.items():
        print(f"{result['status']} {server}")
        if "tools" in result:
            print(f"   └─ {result['tools']} tools available")

    return all(r["status"] == "✅ OK" for r in results.values())


if __name__ == "__main__":
    try:
        from tools.simple_mcp_client import SimpleMCPClient
    except ImportError:
        print("❌ Error: tools.simple_mcp_client not found.")
        print(
            "Make sure you are running this script from the project root (c:\\K3_Firehose)."
        )
        sys.exit(1)

    success = test_all_servers()
    sys.exit(0 if success else 1)
