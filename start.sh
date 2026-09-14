#!/usr/bin/env bash
# One-command startup: install Docker if missing (Linux), prep .env,
# build+run, wait for health, open browser. Safe to re-run: does whatever
# step is still needed and skips the rest. No host Python/Node required.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BOLD='\033[1m'; DIM='\033[2m'; GREEN='\033[32m'; RED='\033[31m'; YELLOW='\033[33m'; RESET='\033[0m'
ok()   { printf "  ${GREEN}✔${RESET} %s\n" "$1"; }
err()  { printf "  ${RED}✘${RESET} %s\n" "$1"; }
step() { printf "${BOLD}%s${RESET}\n" "$1"; }

URL="http://localhost:8000"
HEALTH="$URL/health"

step "Open to Work: startup"
echo

# 1. Docker: install it if missing (Linux only; unattended, uses sudo),
# point at the download page on macOS/Windows since Docker Desktop has no
# unattended installer.
if ! command -v docker >/dev/null 2>&1; then
  case "$(uname -s)" in
    Linux)
      step "Docker not found. Installing via get.docker.com (needs sudo)..."
      curl -fsSL https://get.docker.com | sudo sh
      sudo systemctl enable --now docker >/dev/null 2>&1 || sudo service docker start >/dev/null 2>&1 || true
      sudo usermod -aG docker "$USER" >/dev/null 2>&1 || true
      ok "Docker installed"
      ;;
    Darwin)
      err "Docker not found. macOS needs Docker Desktop (no unattended installer);"
      err "opening the download page. Install it, then re-run this script."
      open "https://www.docker.com/products/docker-desktop/" >/dev/null 2>&1 || true
      exit 1
      ;;
    *)
      err "Docker not found. Auto-install only handled for Linux/macOS here."
      err "Install manually: https://docs.docker.com/get-docker/"
      exit 1
      ;;
  esac
fi

# Freshly installed on Linux means the current shell isn't in the `docker`
# group yet (takes a new login to apply), so fall back to sudo for this run
# only instead of requiring a log out and back in.
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if sudo docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
    err "Using 'sudo docker' for this run. Log out/in once to use docker without sudo from now on."
  else
    err "Docker installed but daemon not reachable. Start it and re-run this script."
    exit 1
  fi
fi
ok "Docker running"

# 2. .env from template, first run only; never overwrite an existing one
if [ ! -f .env ]; then
  cp .env.example .env
  ok "Created .env from .env.example (add GITHUB_TOKEN there later if needed)"
else
  ok ".env present"
fi

# 3. Build + start, detached, quieter output than a raw `up --build`
step "Building image (first run takes a few minutes: downloads Python, torch, models)..."
if ! "${DOCKER[@]}" compose up --build -d --wait 2>&1 | grep -Ev '^\s*$'; then
  echo
  err "docker compose up failed, see log above."
  exit 1
fi
echo

# 4. Poll /health as a second check (compose --wait already waited on the
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

# 5. Best-effort auto-open (never fail the script over this)
if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then open "$URL" >/dev/null 2>&1 &
fi
