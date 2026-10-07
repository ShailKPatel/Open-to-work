# Windows startup without WSL: check Docker Desktop, build and run, wait for
# health, open the browser. Run it through start.cmd (double-click, or
# `.\start.cmd` in a terminal), which bypasses PowerShell's script policy.
# Uses the default port 8000; start.sh (WSL, macOS, Linux) is the one that
# picks a free port when it is busy.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$InstallUrl = "https://docs.docker.com/desktop/setup/install/windows-install/"
$Url = "http://localhost:8000"
# Probe IPv4 directly: compose binds 127.0.0.1 only, and Windows PowerShell
# tries localhost's ::1 first and times out instead of falling back.
$Health = "http://127.0.0.1:8000/health"

function Ok($msg) { Write-Host "  [ok] $msg" -ForegroundColor Green }
function Err($msg) { Write-Host "  [x] $msg" -ForegroundColor Red }

Write-Host "Open to Work: startup" -ForegroundColor White
Write-Host ""

# 1. Docker Desktop has no unattended installer, so open its install page.
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Err "Docker not found. Install Docker Desktop for Windows, open it, then run this again."
    Err "Opening the install page: $InstallUrl"
    Start-Process $InstallUrl
    exit 1
}

# Native stderr under "Stop" is a terminating error in Windows PowerShell 5.1,
# so relax it while probing; the exit code is what decides.
$ErrorActionPreference = "Continue"
docker info *> $null
$DockerExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($DockerExit -ne 0) {
    Err "Docker is installed but not running. Open Docker Desktop, wait until it"
    Err "says Docker is running, then run this again."
    exit 1
}
Ok "Docker running"

# `up --wait` needs Compose v2.1 or newer.
$ComposeVersion = (docker compose version --short 2>$null) -replace '^v', ''
$Parts = "$ComposeVersion".Split('.')
$Major = 0; $Minor = 0
if ($LASTEXITCODE -ne 0 -or $Parts.Count -lt 2 -or
    -not [int]::TryParse($Parts[0], [ref]$Major) -or
    -not [int]::TryParse($Parts[1], [ref]$Minor) -or
    $Major -lt 2 -or ($Major -eq 2 -and $Minor -lt 1)) {
    Err "Docker Compose v2.1 or newer is required. Update Docker Desktop and run this again."
    exit 1
}
Ok "Docker Compose $ComposeVersion"

# 2. .env holds only the optional GITHUB_TOKEN; created once, never overwritten.
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Ok "Created .env (optional: add a GITHUB_TOKEN there for higher GitHub limits)"
}

# 3. Build and start.
Write-Host "Building image (first run takes a few minutes: downloads Python, torch, models)..."
docker compose up --build -d --wait
if ($LASTEXITCODE -ne 0) {
    Err "docker compose up failed, see log above. If port 8000 is in use,"
    Err "free it or run start.sh from WSL, which picks a free port."
    exit 1
}

# 4. Confirm the app answers from outside the container.
Write-Host "Waiting for app to answer..."
$Up = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        Invoke-WebRequest -Uri $Health -UseBasicParsing -TimeoutSec 2 | Out-Null
        $Up = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $Up) {
    Err "App didn't come up in 60s. Check logs: docker compose logs -f app"
    exit 1
}

Ok "App is up"
Write-Host "Ready: $Url" -ForegroundColor Green
Write-Host "Stop with: docker compose down"
Start-Process $Url
