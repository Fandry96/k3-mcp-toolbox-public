import os
import sys
import subprocess
import json
import time
from dotenv import load_dotenv

# Load Env
load_dotenv()

# Configuration
TOOLS = {
    "Mariner (DuckDuckGo)": {
        "image": "68eb20db6109",
        "env": {"PENDING_REQUESTS_LIMIT": "1000"},
        "transport": "stdio",
    },
    "DeepWiki (GitHub)": {
        "image": "26c7b1b8ac6e",
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get(
                "GITHUB_PERSONAL_ACCESS_TOKEN"
            )
        },
        "transport": "stdio",
    },
    "Model Foundry (HF)": {
        "image": "hf-mcp-server:latest",
        "env": {"TRANSPORT": "stdio", "DEFAULT_HF_TOKEN": os.environ.get("HF_TOKEN")},
        "transport": "stdio",
    },
    "Sentinel (Sentry)": {
        "image": "mcp/sentry",
        "env": {"SENTRY_AUTH_TOKEN": os.environ.get("SENTRY_AUTH_TOKEN")},
        "transport": "stdio",
    },
}


def probe_container(name, config):
    print(f"Testing {name}...")

    # 1. Check Image matches
    # (Skipping explicit `docker images` check for speed, relying on run failure)

    # 2. Prepare Command
    cmd = ["docker", "run", "-i", "--rm", "--init"]

    # Add Env Vars
    if config["env"]:
        for k, v in config["env"].items():
            if not v:
                print(f"  ⚠️  WARNING: Missing Env Var for {k}")
            cmd.extend(["-e", f"{k}={v}"])

    cmd.append(config["image"])

    # 3. Launch & Handshake
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
        )

        # JSON-RPC Initialize
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "K3-Ops-Probe", "version": "1.0"},
            },
            "id": 1,
        }

        # Send
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()

        # Read Response (with timeout)
        # Simple read loop
        start = time.time()
        output = ""
        while time.time() - start < 15:
            line = process.stdout.readline()
            if line:
                output = line
                break
            time.sleep(0.1)

        # Cleanup
        process.terminate()
        try:
            process.wait(timeout=2)
        except:
            process.kill()

        if output:
            try:
                data = json.loads(output)
                if "result" in data:
                    print(
                        f"  ✅ ONLINE ({data['result']['serverInfo']['name']} v{data['result']['serverInfo']['version']})"
                    )
                    return True
                else:
                    print(f"  ❌ INVALID RESPONSE: {output[:100]}...")
            except:
                print(f"  ❌ MALFORMED JSON: {output[:100]}...")
        else:
            print("  ❌ TIMEOUT (No Response)")
            # Check stderr
            err = process.stderr.read()
            if err:
                print(f"  [STDERR]: {err[:200]}...")

    except Exception as e:
        print(f"  ❌ EXECUTION ERROR: {e}")

    return False


def main():
    print("🛸 K3 SYSTEM OPERATIONS CHECK 🛸")
    print("================================")

    # Verify .env loaded
    if not os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") or not os.environ.get(
        "HF_TOKEN"
    ):
        print("⚠️  CRITICAL: .env secrets not loaded or missing!")
    else:
        print("🔐 Secrets Loaded: OK")

    print("--------------------------------")

    results = {}
    for name, config in TOOLS.items():
        results[name] = probe_container(name, config)
        print("--------------------------------")

    success_count = sum(1 for v in results.values() if v)
    print(f"\n📊 STATUS: {success_count}/{len(TOOLS)} Systems Online")


if __name__ == "__main__":
    main()
