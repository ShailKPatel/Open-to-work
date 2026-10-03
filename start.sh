#!/usr/bin/env bash
# One-command startup: install Docker if missing (Linux), pick free ports,
# build+run, wait for health, open browser. Safe to re-run: does whatever
# step is still needed and skips the rest. No host Python/Node required.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BOLD='\033[1m'; DIM='\033[2m'; GREEN='\033[32m'; RED='\033[31m'; YELLOW='\033[33m'; RESET='\033[0m'
ok()   { printf "  ${GREEN}✔${RESET} %s\n" "$1"; }
err()  { printf "  ${RED}✘${RESET} %s\n" "$1"; }
step() { printf "${BOLD}%s${RESET}\n" "$1"; }

step "Open to Work: startup"
echo

# 1. Docker: install it if missing on Linux (unattended, uses sudo). Docker
# Desktop on macOS and Windows has no unattended installer, so there the
# install page for that OS is opened instead.
MAC_INSTALL_URL="https://docs.docker.com/desktop/setup/install/mac-install/"
WINDOWS_INSTALL_URL="https://docs.docker.com/desktop/setup/install/windows-install/"
LINUX_INSTALL_URL="https://docs.docker.com/engine/install/"

case "$(uname -s)" in
  Darwin) PLATFORM=mac ;;
  MINGW*|MSYS*|CYGWIN*) PLATFORM=windows ;;
  Linux) grep -qi microsoft /proc/version 2>/dev/null && PLATFORM=wsl || PLATFORM=linux ;;
  *) PLATFORM=other ;;
esac

# Best effort: never fail the script over opening a page.
open_url() {
  case "$PLATFORM" in
    mac) open "$1" ;;
    windows) start "" "$1" ;;
    wsl) explorer.exe "$1" || cmd.exe /c start "" "$1" ;;
    *) xdg-open "$1" ;;
  esac >/dev/null 2>&1 &
}

if ! command -v docker >/dev/null 2>&1; then
  case "$PLATFORM" in
    linux)
      step "Docker not found. Installing via get.docker.com (needs sudo)..."
      if ! curl -fsSL https://get.docker.com | sudo sh; then
        err "Automatic install does not support this Linux distribution."
        err "Install Docker yourself, then re-run this script: $LINUX_INSTALL_URL"
        exit 1
      fi
      sudo systemctl enable --now docker >/dev/null 2>&1 || sudo service docker start >/dev/null 2>&1 || true
      sudo usermod -aG docker "$USER" >/dev/null 2>&1 || true
      ok "Docker installed"
      ;;
    mac)
      err "Docker not found. Install Docker Desktop for Mac, open it, then re-run this script."
      err "Opening the install page: $MAC_INSTALL_URL"
      open_url "$MAC_INSTALL_URL"
      exit 1
      ;;
    windows|wsl)
      err "Docker not found. Install Docker Desktop for Windows, open it, then re-run this script."
      [ "$PLATFORM" = wsl ] && err "If it is already installed, turn on Settings > Resources > WSL integration for this distro."
      err "Opening the install page: $WINDOWS_INSTALL_URL"
      open_url "$WINDOWS_INSTALL_URL"
      exit 1
      ;;
    *)
      err "Docker not found. Install it, then re-run this script: https://docs.docker.com/get-docker/"
      exit 1
      ;;
  esac
fi

# Docker Desktop (macOS, Windows, WSL) installed but not started: say so
# instead of trying sudo, which cannot help there.
if [ "$PLATFORM" != linux ] && ! docker info >/dev/null 2>&1; then
  err "Docker is installed but not running. Open Docker Desktop, wait until it"
  err "says Docker is running, then re-run this script."
  exit 1
fi

# Freshly installed on Linux means the current shell isn't in the `docker`
# group yet (takes a new login to apply), so fall back to sudo for this run
# only instead of requiring a log out and back in. sudo drops environment
# variables by default, so the chosen ports are passed through explicitly.
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if sudo docker info >/dev/null 2>&1; then
    DOCKER=(sudo --preserve-env=APP_PORT,QDRANT_PORT docker)
    err "Using 'sudo docker' for this run. Log out/in once to use docker without sudo from now on."
  else
    err "Docker installed but daemon not reachable. Start it and re-run this script."
    exit 1
  fi
fi
ok "Docker running"

# `up --wait` below needs Compose v2.1 or newer, run as `docker compose`.
# The old standalone `docker-compose` (v1) is not enough.
COMPOSE_VERSION="$("${DOCKER[@]}" compose version --short 2>/dev/null || true)"
COMPOSE_VERSION="${COMPOSE_VERSION#v}"
COMPOSE_MAJOR="${COMPOSE_VERSION%%.*}"
COMPOSE_MINOR="$(echo "$COMPOSE_VERSION" | cut -d. -f2)"
if ! [[ "$COMPOSE_MAJOR" =~ ^[0-9]+$ && "$COMPOSE_MINOR" =~ ^[0-9]+$ ]] \
  || [ "$COMPOSE_MAJOR" -lt 2 ] \
  || { [ "$COMPOSE_MAJOR" -eq 2 ] && [ "$COMPOSE_MINOR" -lt 1 ]; }; then
  err "Docker Compose v2.1 or newer is required (found: ${COMPOSE_VERSION:-none})."
  err "Update Docker or install the Compose plugin: https://docs.docker.com/compose/install/"
  exit 1
fi
ok "Docker Compose $COMPOSE_VERSION"

# 2. .env only holds the optional GITHUB_TOKEN. Created on first run so
# there is an obvious place to put it; never overwritten.
if [ ! -f .env ]; then
  cp .env.example .env
  ok "Created .env (optional: add a GITHUB_TOKEN there for higher GitHub limits)"
fi

# 3. Ports: 8000 for the app and 6333 for Qdrant, or the next free port
# when something else already holds one. A port this app's own running
# containers publish is kept, so a re-run doesn't move the app.
port_in_use() { (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1; }
free_port() {
  local port=$1
  while port_in_use "$port"; do port=$((port + 1)); done
  echo "$port"
}
published_port() {
  "${DOCKER[@]}" compose port "$1" "$2" 2>/dev/null | awk -F: 'NF { print $NF; exit }' || true
}

APP_PORT="$(published_port app 8000)"
APP_PORT="${APP_PORT:-$(free_port 8000)}"
QDRANT_PORT="$(published_port qdrant 6333)"
QDRANT_PORT="${QDRANT_PORT:-$(free_port 6333)}"
export APP_PORT QDRANT_PORT
[ "$APP_PORT" = 8000 ] && ok "App port 8000" || ok "Port 8000 is busy, using $APP_PORT for the app"
[ "$QDRANT_PORT" = 6333 ] && ok "Qdrant port 6333" || ok "Port 6333 is busy, using $QDRANT_PORT for Qdrant"

URL="http://localhost:$APP_PORT"
HEALTH="$URL/health"

# 4. Build + start, detached, quieter output than a raw `up --build`
step "Building image (first run takes a few minutes: downloads Python, torch, models)..."
if ! "${DOCKER[@]}" compose up --build -d --wait 2>&1 | grep -Ev '^\s*$'; then
  echo
  err "docker compose up failed, see log above."
  exit 1
fi
echo

# 5. Poll /health as a second check (compose --wait already waited on the
# container healthcheck, but confirm the port answers from outside).
step "Waiting for app to answer..."
SPIN='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
i=0
for _ in $(seq 1 60); do
  if curl -sf "$HEALTH" >/dev/null 2>&1; then
    printf "\r  ${GREEN}✔${RESET} App is up                     \n"
    break
  fi
  i=$(( (i + 1) % 10 ))
  printf "\r  ${YELLOW}${SPIN:$i:1}${RESET} waiting for %s" "$HEALTH"
  sleep 1
done
echo

if ! curl -sf "$HEALTH" >/dev/null 2>&1; then
  err "App didn't come up in 60s. Check logs: docker compose logs -f app"
  exit 1
fi

echo -e "${BOLD}${GREEN}Ready →${RESET} ${BOLD}$URL${RESET}"
echo -e "${DIM}Stop with: docker compose down${RESET}"
echo

# 6. Best-effort auto-open
open_url "$URL"
