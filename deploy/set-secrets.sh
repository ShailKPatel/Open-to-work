#!/bin/bash
# Sends a local env file to the VM as /opt/otw/.env, readable by root only.
# Run from your machine after `az login`. Keep the file out of git.
#   deploy/set-secrets.sh <resource-group> <vm-name> <env-file>
set -euo pipefail
group=$1 vm=$2 file=$3
encoded=$(base64 -w0 "$file")
script=$(mktemp)
trap 'rm -f "$script"' EXIT
cat > "$script" <<SCRIPT
set -euo pipefail
cd /opt/otw
touch .env && chmod 600 .env
image=\$(grep '^IMAGE=' .env || true)
echo '$encoded' | base64 -d > .env
[ -z "\$image" ] || echo "\$image" >> .env
echo SECRETS_OK
SCRIPT
az vm run-command invoke -g "$group" -n "$vm" --command-id RunShellScript \
  --scripts @"$script" --query 'value[0].message' -o tsv | grep -q SECRETS_OK
echo "secrets written"
