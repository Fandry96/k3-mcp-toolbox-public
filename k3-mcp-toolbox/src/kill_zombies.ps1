# K3-9000 Ops Script: Zombie Process Mitigation
# Scope: Chrome, ChromeDriver, Edge, EdgeDriver
# Action: Force Kill (/F) if hung

$targets = @("chrome", "chromedriver", "msedge", "msedgedriver", "python")
$cleaned = 0

Write-Host "--- DETECTING ZOMBIE PROCESSES ---"

foreach ($target in $targets) {
    $processes = Get-Process -Name $target -ErrorAction SilentlyContinue
    if ($processes) {
        $count = $processes.Count
        Write-Host "Found $count instance(s) of $target..."

        # Filter for high-memory/stuck processes if needed, or just nuke them all
        # For Ops hygiene, we assume any lingering driver is a leak.
        Stop-Process -Name $target -Force -ErrorAction SilentlyContinue

        if ($?) {
            Write-Host "  [+] Terminated $target" -ForegroundColor Green
            $cleaned += $count
        } else {
            Write-Host "  [!] Failed to terminate $target (Access Denied?)" -ForegroundColor Red
        }
    }
}

Write-Host "--- OPS COMPLETE ---"
Write-Host "Total processes cleaned: $cleaned"
