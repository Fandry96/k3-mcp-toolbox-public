# kill_zombies.ps1
# Force-closes processes that might be locking the Chrome profile or MCP bridge.

Write-Host "🧟 Hunting for zombie processes..." -ForegroundColor Yellow

# 1. Kill Node processes (often these are the MCP servers)
# We try to be specific if possible, but "node.exe" is generic.
# Ideally we'd filter by command line, but for now, we'll just check for node.
$nodeProcs = Get-Process node -ErrorAction SilentlyContinue
if ($nodeProcs) {
    Write-Host "Found $($nodeProcs.Count) Node processes. Terminating..." -ForegroundColor Red
    Stop-Process -Name node -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "No Node zombies found." -ForegroundColor Green
}

# 2. Kill Chrome processes (headless instances)
$chromeProcs = Get-Process chrome -ErrorAction SilentlyContinue
if ($chromeProcs) {
    Write-Host "Found $($chromeProcs.Count) Chrome processes. Terminating..." -ForegroundColor Red
    Stop-Process -Name chrome -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "No Chrome zombies found." -ForegroundColor Green
}

Write-Host "✨ Cleanup complete. The profiles should be unlocked." -ForegroundColor Cyan
