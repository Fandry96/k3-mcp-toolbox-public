# K3 MRL Watcher Background Daemon Launcher
# Usage: ./start_mrl_watcher.ps1

Write-Host "Starting K3 MRL Auto-Indexing Watcher Daemon..." -ForegroundColor Cyan
$scriptPath = Join-Path $PSScriptRoot "src\k3_mrl_watcher.py"

python $scriptPath
