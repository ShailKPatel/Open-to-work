# Running the public demo

`make start` needs none of this. These variables are read from the process
environment only, never from `.env`. Set the secret ones as secrets on the host.

The image is built by `.github/workflows/release.yml` and published to GHCR. It
needs a persistent volume at `/app/data` and a reachable Qdrant (`QDRANT_URL`).
The app must sit behind HTTPS: session cookies are `Secure`.

| Variable | Secret | Purpose |
|---|---|---|
| `DEMO_MODE=true` | | Turns on sign-in and the route gate. The app will not start if a value below is missing. |
| `DEMO_PUBLIC_HOST` | | Public host name, no scheme, for example `demo.example.com`. |
| `DEMO_GITHUB_CLIENT_ID` | | GitHub OAuth app. Its callback is `https://<host>/auth/callback`. |
| `DEMO_GITHUB_CLIENT_SECRET` | yes | Same OAuth app. |
| `DEMO_SESSION_SECRET` | yes | At least 32 random characters; signs session cookies. |
| `DEMO_ADMIN_GITHUB_ID` | | Your numeric GitHub id. This account skips the gate. |
| `DEMO_SAMPLE_ACCOUNT_ID` | | Optional. An account id visitors can browse read-only from `/welcome` without signing in. Set it after that account exists. |
| `APP_SECRET_KEY` | yes | Fernet key encrypting stored API keys. Must not change. |
| `GITHUB_TOKEN` | yes | Raises the GitHub API limit for syncs. |
| `DEMO_DAILY_RESUME_BUILDS` | | Default 5 per user. |
| `DEMO_DAILY_LLM_CALLS` | | Default 200 per user. |
| `DEMO_DAILY_COST_USD` | | Default 0.50 per user. |
| `DEMO_MAX_ACCOUNTS` | | Default 50. |
| `DEMO_DAILY_WRITES`, `DEMO_DAILY_INTAKE` | | Per user. Defaults 500 changes and 30 uploads or re-reads. |
| `DEMO_MAX_SOURCES`, `DEMO_MAX_RUNNING_BUILDS` | | Defaults 5 per user and 3 in total. |
| `TRACING_ENABLED=true` | | With `LANGSMITH_API_KEY` (secret), sends traces, including resume content, to LangSmith. |
| `MCP_GITHUB_ENABLED=true` | | With `GITHUB_TOKEN`, the profile agent also reads each profile's own repos through GitHub's remote MCP server (read-only, allowlisted tools, own repos only). The token is the deployment's, so leave it off unless that is acceptable. `MCP_GITHUB_URL` overrides the endpoint. |

## Azure VM

`infra/azure/main.bicep` creates one VM (default `Standard_B2als_v2`, 2 vCPU,
4 GiB), a separate data disk, HTTPS-only networking and the identity GitHub
deploys as. Run these from a machine with the Azure CLI:

1. Find a region your subscription allows. Azure for Students sets a list per
   account: Portal, Policy, Assignments, "Allowed resource deployment regions".
   Then confirm the size is offered there:
   `az vm list-skus -l <region> --size Standard_B2als_v2 --all -o table`
   (an entry under Restrictions means pick another size or region).
2. `az group create -n otw-demo -l <region>`
3. `az deployment group create -g otw-demo -f infra/azure/main.bicep -p location=<region> dnsLabel=<name> githubRepo=<owner/repo> adminPublicKey="$(cat ~/.ssh/id_ed25519.pub)"`
   and note the outputs.
4. Create a GitHub OAuth app with callback `https://<publicHost>/auth/callback`.
5. Write a local env file with the variables in the table above (never commit
   it), then `deploy/set-secrets.sh otw-demo <vmName> <file>`.
6. In the GitHub repository, add variables `AZURE_CLIENT_ID` (deployerClientId),
   `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP` and
   `AZURE_VM_NAME`. Merge to main. After the first image is pushed, make the
   GHCR package public so the VM can pull it; it contains no secrets.

## Any Ubuntu server (VPS)

Any Ubuntu 22.04 or 24.04 server with 4 GB of RAM works the same way; data
lives on its own disk in `/mnt/otw`, which survives reboots and redeploys.

1. Make a key pair for GitHub Actions: `ssh-keygen -t ed25519 -N "" -f otw-deploy`.
2. Prepare the server once, as root:
   `ssh root@<ip> 'bash -s' < deploy/server-setup.sh "$(cat otw-deploy.pub)"`
   (it prints `SETUP_OK`). SSH passwords are turned off and only 22, 80 and
   443 stay open.
3. Point a host name at the IP. Without a domain, `<ip-with-dashes>.sslip.io`
   works, for example `203-0-113-5.sslip.io`.
4. Create the GitHub OAuth app and the env file as for Azure (steps 4 and 5),
   then `deploy/set-secrets-ssh.sh deploy@<ip> <file>`.
5. In the GitHub repository add the secret `DEPLOY_SSH_KEY` (the contents of
   `otw-deploy`) and the variables `DEPLOY_SSH_HOST` (the IP) and
   `DEPLOY_SSH_KNOWN_HOSTS` (the output of `ssh-keyscan <ip>`). The next merge
   to main deploys; to deploy the current main now, re-run the latest Release
   run.

## Restoring

Run with the app stopped. It downloads a backup, moves the current files
aside (nothing is deleted), and rebuilds the search index from the database,
which is why Qdrant is never backed up. It needs the original `APP_SECRET_KEY`.

```
cd /opt/otw
docker compose stop app
docker compose run --rm app python -m app.core.restore --list
docker compose run --rm app python -m app.core.restore --yes      # newest, or --key <name>
docker compose up -d
```

To rebuild only the search index: `docker compose exec app python -m app.retrieval.rebuild --fresh`.

## Notes

**Certificates.** `*.cloudapp.azure.com` names can hit Let's Encrypt's shared
limit. Caddy retries and falls back to ZeroSSL. If that also fails, point any
domain you own at the VM's IP and change `DEMO_PUBLIC_HOST`.

**Backups.** Create a private bucket on Backblaze B2 or Cloudflare R2 and add
`BACKUP_BUCKET`, `BACKUP_ENDPOINT`, `BACKUP_REGION`, `BACKUP_ACCESS_KEY_ID` and
`BACKUP_SECRET_ACCESS_KEY` to the env file. Cron runs it nightly and keeps 7.

**Cost.** On Azure, about $18 a month for the VM in Central India, plus the
disks and the static IP: roughly $25 to $28 a month, so $100 lasts three to
four months running nonstop. Set a budget alert. Stopping
(deallocating) the VM ends compute charges; disks still bill.

After the first deploy, sign in with the admin account, add your LLM keys on
`/apis`, and set the models and monthly budget. Keys belong to the account
that adds them: every visitor adds their own, and nobody else's work uses them.

**Sample profile.** To let people look around without signing in, use your own
account: set placeholder contact details on it (Portfolio, Contact), build a
resume or two, then set `DEMO_SAMPLE_ACCOUNT_ID` to its id and resend the env
file. The id is in the address bar of any portfolio API call, or run
`docker compose exec app python -c "from app.core.db import Account, get_db; print([(a.id, a.github_username) for a in get_db().query(Account)])"`
on the VM. Visitors can read everything on it except its keys and the
original uploaded resume file, and can change nothing.

**Storage.** SQLite WAL mode, which the app database and the graph checkpointer
both use, needs shared memory and is unsafe on network file systems (SMB, NFS,
EFS, Azure Files). Use a block volume, one instance only.

**Usage numbers:** `DATABASE_URL=... python -m scripts.usage_report`.
