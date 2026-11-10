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

**Certificates.** `*.cloudapp.azure.com` names can hit Let's Encrypt's shared
limit. Caddy retries and falls back to ZeroSSL. If that also fails, point any
domain you own at the VM's IP and change `DEMO_PUBLIC_HOST`.

**Backups.** Create a private bucket on Backblaze B2 or Cloudflare R2 and add
`BACKUP_BUCKET`, `BACKUP_ENDPOINT`, `BACKUP_REGION`, `BACKUP_ACCESS_KEY_ID` and
`BACKUP_SECRET_ACCESS_KEY` to the env file. Cron runs it nightly and keeps 7.

**Cost.** About $18 a month for the VM in Central India plus the disks, so
$100 lasts roughly five months running nonstop. Set a budget alert. Stopping
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
