#!/usr/bin/env python3
"""
k3-agent-ops — Windows Runtime Sentinel MCP Server
Provides process zombie reaping, port conflict diagnosis and freeing,
and system health telemetry to autonomous agents without requiring raw shell grants.
"""

import os
import json
import subprocess
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("k3-agent-ops")

CURRENT_PID = os.getpid()


def _run_powershell(command: str) -> str:
    """Executes a PowerShell command with UTF-8 encoding and error capture."""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        return proc.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Error: PowerShell command timed out (20s)."
    except Exception as e:
        return f"Error executing PowerShell: {e}"


@mcp.tool()
def ops_kill_zombies() -> str:
    """
    Terminates orphaned and runaway background processes (chromedriver, msedgedriver,
    stale chrome/edge test instances) while safely preserving active IDE and agent runtime PIDs.
    """
    reaped = []
    skipped = []

    # Get our own PID and parent PID to prevent suicide
    safe_pids = {CURRENT_PID, os.getppid()}

    # Target only test drivers and headless automation instances (protects regular user browser windows)
    cmd = (
        "Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | "
        "Where-Object { "
        "  ($_.ProcessName -in 'chromedriver.exe','msedgedriver.exe') -or "
        "  (($_.ProcessName -in 'chrome.exe','msedge.exe') -and ($_.CommandLine -match '(--headless|--test-type|--remote-debugging-port)')) "
        "} | "
        "Select-Object @{Name='Id';Expression={$_.ProcessId}}, ProcessName, @{Name='WorkingSet64';Expression={$_.WorkingSetSize}} | "
        "ConvertTo-Json"
    )
    raw = _run_powershell(cmd)
    if not raw or raw == "null":
        return "Clean: No zombie browser or driver processes detected."

    try:
        data = json.loads(raw)
        procs = [data] if isinstance(data, dict) else data
    except Exception:
        # Fallback to driver-only Stop-Process
        kill_cmd = "Stop-Process -Name chromedriver,msedgedriver -Force -ErrorAction SilentlyContinue"
        _run_powershell(kill_cmd)
        return "Cleaned: Terminated any hung chromedriver / msedgedriver instances."

    for p in procs:
        pid = p.get("Id")
        name = p.get("ProcessName", "unknown")
        ram_mb = p.get("WorkingSet64", 0) / (1024 * 1024)

        if pid in safe_pids:
            skipped.append(f"{name} (PID {pid}) [Self/Parent Protected]")
            continue

        try:
            # Terminate target
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5)
            reaped.append(f"{name} (PID {pid}, {ram_mb:.1f} MB RAM)")
        except Exception as e:
            skipped.append(f"{name} (PID {pid}, Error: {e})")

    lines = ["=== K3 Ops Zombie Reaper Report ==="]
    if reaped:
        lines.append(f"Successfully Reaped ({len(reaped)}):")
        for r in reaped:
            lines.append(f"  [+] {r}")
    else:
        lines.append("No actionable zombie processes required termination.")

    if skipped:
        lines.append(f"\nPreserved / Skipped ({len(skipped)}):")
        for s in skipped:
            lines.append(f"  [-] {s}")

    return "\n".join(lines)


@mcp.tool()
def ops_check_ports(ports: Optional[List[int]] = None) -> str:
    """
    Scans specified TCP ports for active listeners and identifies the owning process and PID.
    Useful for resolving Errno 10048 (port conflict) before starting dev servers.
    Args:
        ports: List of port numbers to probe. Default: [3000, 5173, 8080, 8081, 9222, 26646]
    """
    if not ports:
        ports = [3000, 5173, 8080, 8081, 9222, 26646]

    try:
        valid_ports = [int(p) for p in ports if 1 <= int(p) <= 65535]
    except (ValueError, TypeError):
        return "Error: ports must be a list of integer port numbers between 1 and 65535."

    if not valid_ports:
        return "Error: No valid port numbers provided (must be between 1 and 65535)."

    ports_str = ",".join(str(p) for p in valid_ports)
    cmd = (
        f"$targetPorts = @({ports_str}); "
        "Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | "
        "Where-Object { $_.LocalPort -in $targetPorts } | "
        "ForEach-Object { "
        "  $proc = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue; "
        "  [PSCustomObject]@{ "
        "    Port = $_.LocalPort; "
        "    PID = $_.OwningProcess; "
        "    Process = if ($proc) { $proc.ProcessName } else { 'Unknown' }; "
        "    Address = $_.LocalAddress "
        "  } "
        "} | ConvertTo-Json"
    )

    raw = _run_powershell(cmd)
    if not raw or raw == "null":
        return f"All probed ports ({ports_str}) are FREE and available."

    try:
        data = json.loads(raw)
        items = [data] if isinstance(data, dict) else data
    except Exception:
        return f"Raw port scan output:\n{raw}"

    lines = ["=== Port Status Scan ==="]
    occupied_ports = set()
    for item in items:
        port = item.get("Port")
        pid = item.get("PID")
        proc_name = item.get("Process")
        addr = item.get("Address", "0.0.0.0")
        occupied_ports.add(port)
        lines.append(f"  [BUSY] Port {port:<5} -> PID {pid:<6} ({proc_name}) on {addr}")

    free_ports = [p for p in ports if p not in occupied_ports]
    if free_ports:
        lines.append(f"\nAvailable Ports ({len(free_ports)}): {', '.join(str(p) for p in free_ports)}")

    return "\n".join(lines)


@mcp.tool()
def ops_free_port(port: int) -> str:
    """
    Frees an occupied TCP port by terminating the owning process.
    Refuses to kill the current agent or parent process.
    Args:
        port: The TCP port to free (e.g. 3000, 8080, 5173)
    """
    if port < 1 or port > 65535:
        return f"Invalid port number: {port}"

    cmd = (
        f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty OwningProcess -First 1"
    )
    raw = _run_powershell(cmd)
    if not raw or not raw.strip().isdigit():
        return f"Port {port} is not in LISTEN state (already free)."

    target_pid = int(raw.strip())
    if target_pid in {CURRENT_PID, os.getppid()}:
        return f"BLOCKED: Port {port} is held by current agent process (PID {target_pid}). Suicide prevented."

    # Identify process name
    proc_name = _run_powershell(f"(Get-Process -Id {target_pid} -ErrorAction SilentlyContinue).ProcessName") or "Unknown"

    # Terminate process
    kill_res = subprocess.run(["taskkill", "/F", "/PID", str(target_pid)], capture_output=True, text=True)
    if kill_res.returncode == 0:
        return f"SUCCESS: Terminated process '{proc_name}' (PID {target_pid}). Port {port} is now free."
    else:
        return f"FAILED to kill PID {target_pid}: {kill_res.stderr.strip()}"


@mcp.tool()
def ops_system_health() -> str:
    """
    Returns real-time Windows system diagnostics: CPU load, RAM usage, C: disk space,
    and agent process counts (Python, Node, PowerShell).
    """
    cmd = (
        "$os = Get-CimInstance Win32_OperatingSystem; "
        "$cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average; "
        "$disk = Get-PSDrive C; "
        "$py = (Get-Process python -ErrorAction SilentlyContinue).Count; "
        "$node = (Get-Process node -ErrorAction SilentlyContinue).Count; "
        "$ps = (Get-Process pwsh,powershell -ErrorAction SilentlyContinue).Count; "
        "[PSCustomObject]@{ "
        "  CPU_Pct = [math]::Round($cpu, 1); "
        "  Total_RAM_GB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2); "
        "  Free_RAM_GB = [math]::Round($os.FreePhysicalMemory / 1MB, 2); "
        "  Disk_Free_GB = [math]::Round($disk.Free / 1GB, 1); "
        "  Disk_Total_GB = [math]::Round(($disk.Used + $disk.Free) / 1GB, 1); "
        "  Python_Count = $py; "
        "  Node_Count = $node; "
        "  PowerShell_Count = $ps "
        "} | ConvertTo-Json"
    )
    raw = _run_powershell(cmd)
    try:
        d = json.loads(raw)
        total_ram = d.get("Total_RAM_GB", 0)
        free_ram = d.get("Free_RAM_GB", 0)
        used_ram = total_ram - free_ram
        ram_pct = (used_ram / total_ram * 100) if total_ram else 0
        disk_free = d.get("Disk_Free_GB", 0)
        disk_total = d.get("Disk_Total_GB", 0)
        disk_pct = ((disk_total - disk_free) / disk_total * 100) if disk_total else 0

        lines = [
            "=== K3 Windows System Health ===",
            f"CPU Load: {d.get('CPU_Pct', 0)}%",
            f"RAM: {used_ram:.1f} GB / {total_ram:.1f} GB ({ram_pct:.1f}% used) | Free: {free_ram:.1f} GB",
            f"Disk (C:): {disk_total - disk_free:.1f} GB / {disk_total:.1f} GB ({disk_pct:.1f}% used) | Free: {disk_free:.1f} GB",
            "\nAgent Process Footprint:",
            f"  - Python instances    : {d.get('Python_Count', 0)}",
            f"  - Node.js instances   : {d.get('Node_Count', 0)}",
            f"  - PowerShell instances: {d.get('PowerShell_Count', 0)}",
        ]
        return "\n".join(lines)
    except Exception:
        return f"Raw diagnostics:\n{raw}"


if __name__ == "__main__":
    mcp.run()
