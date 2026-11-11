#!/bin/bash
# Sends a local env file to a plain server as /opt/otw/.env, readable only
# by the deploy user, keeping the IMAGE line the last deploy wrote. The
# SSH twin of deploy/set-secrets.sh (Azure). Keep the file out of git.
#   deploy/set-secrets-ssh.sh deploy@<ip> <env-file>
# Then restart the app on the server to pick it up:
#   ssh deploy@<ip> 'cd /opt/otw && docker compose up -d'
set -euo pipefail
target=${1:?usage: set-secrets-ssh.sh deploy@<ip> <env-file>}
file=${2:?usage: set-secrets-ssh.sh deploy@<ip> <env-file>}
ssh "$target" 'set -euo pipefail
cd /opt/otw
umask 077
image=$(grep "^IMAGE=" .env 2>/dev/null || true)
cat > .env.new
[ -z "$image" ] || echo "$image" >> .env.new
mv .env.new .env
echo SECRETS_OK' < "$file" | grep -q SECRETS_OK
echo "secrets written"
