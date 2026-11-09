#!/bin/bash
# Runs on the VM through `az vm run-command`, with IMAGE, COMPOSE_B64 and
# CADDY_B64 set by the lines the deploy workflow puts in front of this file.
# Prints DEPLOY_OK last: run-command reports success whatever a script exits
# with, so that marker is how the workflow knows.
set -euo pipefail
cd /opt/otw
test -f .env || { echo "no /opt/otw/.env: run deploy/set-secrets.sh first"; exit 1; }
echo "$COMPOSE_B64" | base64 -d > docker-compose.yml
echo "$CADDY_B64" | base64 -d > Caddyfile
# Compose reads IMAGE from .env; everything else in that file stays.
sed -i '/^IMAGE=/d' .env
echo "IMAGE=$IMAGE" >> .env
docker compose pull --quiet
docker compose up -d --remove-orphans
for _ in $(seq 1 60); do
  if docker compose exec -T app curl -fs http://localhost:8000/health >/dev/null 2>&1; then
    docker image prune -f >/dev/null
    echo DEPLOY_OK
    exit 0
  fi
  sleep 5
done
docker compose logs --tail 40 app
exit 1
