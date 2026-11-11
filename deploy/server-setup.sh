#!/bin/bash
# One-time setup of any Ubuntu 22.04/24.04 server for the public demo, the
# plain-VPS twin of infra/azure/cloud-init.yaml. Run once as root:
#
#   ssh root@<ip> 'bash -s' < deploy/server-setup.sh "<deploy public key>"
#
# Installs Docker, makes a `deploy` user that GitHub Actions signs in as
# (key only), turns off SSH passwords, opens only 22/80/443, adds swap and
# the nightly backup. Data lives in /mnt/otw on the server's own disk,
# which survives reboots, container restarts and redeploys.
set -euo pipefail
deploy_key=${1:?usage: server-setup.sh "<ssh public key for GitHub Actions>"}

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -yq docker.io docker-compose-v2 curl ufw
systemctl enable --now docker

id deploy >/dev/null 2>&1 || useradd --create-home --shell /bin/bash deploy
# Membership of the docker group is root-equivalent; this user exists only
# so the key GitHub holds is not root's own.
usermod -aG docker deploy
install -d -m 700 -o deploy -g deploy /home/deploy/.ssh
grep -qxF "$deploy_key" /home/deploy/.ssh/authorized_keys 2>/dev/null \
  || echo "$deploy_key" >> /home/deploy/.ssh/authorized_keys
chown deploy:deploy /home/deploy/.ssh/authorized_keys
chmod 600 /home/deploy/.ssh/authorized_keys

install -d -m 700 -o deploy -g deploy /opt/otw
mkdir -p /mnt/otw/data /mnt/otw/qdrant /mnt/otw/caddy /mnt/otw/hf

# Keys only. A drop-in, so a package upgrade of sshd_config keeps it.
printf 'PasswordAuthentication no\nKbdInteractiveAuthentication no\nPermitRootLogin prohibit-password\n' \
  > /etc/ssh/sshd_config.d/10-otw.conf
systemctl reload ssh || systemctl reload sshd

ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# The embedding model and a LaTeX compile can briefly exceed 4 GiB together.
if [ ! -f /swapfile ]; then
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# Nightly backup; does nothing until BACKUP_BUCKET is set.
echo '30 20 * * * root cd /opt/otw && docker compose exec -T app python -m app.core.backup >> /var/log/otw-backup.log 2>&1' \
  > /etc/cron.d/otw-backup

echo SETUP_OK
