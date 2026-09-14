# Open to Work

A self-hosted job search workspace. It builds a skill profile from GitHub activity and manually entered experience, tracks job postings, and generates tailored LaTeX resumes grounded in that profile.

Each skill links back to the evidence behind it: a dependency manifest, a README, a repository description, or a role. Generated resume content is restricted to projects and skills from that evidence.

## Features

### Profile

- **GitHub sync** from a username, profile URL, or single repository URL. Progress streams to the browser over Server-Sent Events and can be cancelled. When GitHub rate limits a batch partway through, the repositories already fetched are kept and the sync reports how many were saved. READMEs, manifests, and commit stats are refetched only for repositories pushed since the last sync.
- **Skill extraction** per repository. Dependencies come from manifests for Python, JavaScript, Go, Rust, Ruby, and JVM (Maven/Gradle) projects with no LLM involved. One LLM call reads the README, or the repository description if there is no README, and repositories with neither make no LLM call.
- **Evidence weighting** by source type, fork status, commit recency, and commit volume. Evidence for the same skill across repositories is combined with a noisy-OR.
- **Per-repository status** (pending, extracted, failed, rate limited, no signal) with manual reprocessing. A provider rate limit or budget cap stops the batch. The next run continues from there without repeating completed repositories.
- **Manual portfolio data**: projects, work experience as individual bullet points, education, skills, contact details, social links, and starred projects and skills.
- **Resume library**: upload PDF or image resumes. Tags, target roles, a summary, and work history are extracted, then merged into the profile without duplicating existing skills or roles.

### Jobs

- **Four ways to add a posting**: paste text, fetch a public URL, upload a screenshot, or fetch a login-protected page through a stored browser login profile (Playwright).
- **Structured extraction** of salary range, employment type, seniority, required experience, and required skills with expected proficiency.
- **Role families**: titles such as "ML Engineer" and "Machine Learning Engineer" are grouped by embedding similarity. An LLM call is made only when a title has no close existing match.
- **Application tracking** with applied date and notes, a skill gap view for each posting, and skill-demand analytics across all collected postings.

### Resume generation

- For a selected posting, semantic search retrieves candidate projects, skills, and experience bullets. One LLM call then picks from those candidates and writes the summary and project bullets. Any project or skill the model returns that was not a candidate is discarded.
- Work history and education are always included in full and are never sent to the model for editing.
- Output is rendered to LaTeX from one-page or two-page templates and compiled with Tectonic. If the PDF is longer than the page limit, a multimodal pass reads the rendered PDF and removes one item at a time, in a fixed order (a skill first, then a project bullet, then an experience bullet), until the resume fits.
- Generated resumes are saved to the library and can be revised later with a plain-language instruction.

### Operations

- Multiple local profiles on one machine, with no login.
- LLM provider keys are entered in the app and encrypted at rest. Supported providers include OpenAI, Azure OpenAI, AWS Bedrock, Mistral, and Ollama. Keys can be disabled, limited to specific profiles, or given their own monthly budget.
- Every LLM call goes through a single client. The client caches responses by content hash, checks the monthly budget before sending a request, and records cost, tokens, and latency for each call.
- A monitor page shows GitHub rate-limit headroom, LLM spend broken down by provider, key, profile, tier, and model, and a log of rate-limit events.
- The SQLite database is snapshotted to `data/backups/` on every startup, and the 10 most recent snapshots are kept.

## Architecture

```mermaid
flowchart LR
    GH[GitHub API] --> SYNC[ingest/github]
    JP[Job posting<br/>text · URL · screenshot · login] --> JOBS[ingest/jobs]
    SYNC --> PROF[profile<br/>extraction + weighting]
    JOBS --> JEX[profile<br/>job extraction + role families]
    PROF --> DB[(SQLite)]
    JEX --> DB
    DB --> IDX[retrieval<br/>local embeddings]
    IDX --> QD[(Qdrant)]
    QD --> RB[resume_build<br/>select · render · compile · fit]
    DB --> RB
    RB --> PDF[PDF]
    LLM[core/llm<br/>cache · budget · keys] -.-> PROF
    LLM -.-> JEX
    LLM -.-> RB
```

Module responsibilities, the data model, and project conventions are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tech stack

Python 3.12, FastAPI, SQLAlchemy with SQLite, Qdrant, sentence-transformers (`BAAI/bge-base-en-v1.5`, run locally), LiteLLM, Jinja2 with Tailwind CSS and Alpine.js, Tectonic, Playwright, pypdf, rank-bm25, pytest, ruff, and mypy. Runs with Docker Compose.

## Getting started

Requires Docker with Compose.

```bash
git clone <repository-url> open-to-work
cd open-to-work
cp .env.example .env
docker compose up --build
```

Alternatively, `make start` checks for Docker, creates `.env` if it doesn't exist, waits for the health check, and opens the browser.

Open http://localhost:8000, create a profile, add an LLM API key when prompted, and sync a GitHub account.

### Configuration

Settings are read from `.env`. API keys are not set here; they are entered in the app.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GITHUB_TOKEN` | empty | Optional. Raises the GitHub API limit from 60 to 5,000 requests per hour. |
| `LLM_BULK_MODEL` | `openai/gpt-4o-mini` | Per-repository extraction, role-family naming, and groundedness checks. |
| `LLM_QUALITY_MODEL` | `openai/gpt-4o` | Resume generation, page fitting, and job and resume extraction. |
| `MONTHLY_BUDGET_USD` | `20.0` | Cap across all LLM calls, checked before each request. |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Local sentence-transformers model. |
| `APP_SECRET_KEY` | generated | Fernet key for stored credentials. If unset, a key is generated into `data/.secret_key` with 0600 permissions. |
| `QDRANT_URL` | `http://localhost:6333` | Set to the `qdrant` service automatically under Docker Compose. |

Application data (the SQLite database, uploaded files, backups, and the secret key) lives in `./data`, which is mounted into the container.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
docker compose up -d qdrant

make test                              # unit tests; no network or API key required
make coverage                          # unit tests with a line and branch coverage report
make lint                              # ruff and mypy
make test-live                         # opt-in checks against real services, see Testing
make ingest ACCOUNT=<github-username>  # sync from the command line
make eval ACCOUNT=<account-id>         # retrieval and groundedness evaluation
```

### Testing

Tests are written with pytest and live in two suites.

`tests/unit/` runs on every `make test`. It covers each layer on its own: the GitHub client and sync pipeline, job and resume ingestion, skill extraction and weighting, retrieval, the LLM gateway (caching, budgets, key rotation), credential encryption, resume building, the evaluation metrics, and the command-line entry points. Every API router is exercised through FastAPI's test client, and a route sweep renders every page and calls every list endpoint, so a broken template or a failing route is caught even without a dedicated test. Unit tests use temporary SQLite databases, in-memory Qdrant, and injected fakes for GitHub, the LLM, the embedding model, Tectonic, and Playwright. `make coverage` adds a line and branch coverage report and fails below the threshold set in `pyproject.toml`. Compiling PDFs outside Docker requires a local Tectonic install.

`tests/live/` talks to real services and never runs by default. `make test-live` runs every live suite that has its variable set and skips the rest:

| Variable | Suite | Checks |
| --- | --- | --- |
| `LIVE_LLM_API_KEY` (and optionally `LIVE_LLM_PROVIDER`) | `test_llm_live.py` | Text, image, PDF, and multi-turn calls through the LLM gateway. Billed. |
| `LIVE_GITHUB=1` (and optionally `GITHUB_TOKEN`) | `test_github_live.py` | GitHub API reachability, authentication, 404 handling, and a single-repository sync into a temporary database. |
| `LIVE_APP_URL` (for example `http://localhost:8000`) | `test_app_live.py` | Health, every page, JSON endpoints, and the app's own GitHub status check against a running instance. Read-only. |

### Evaluation

`make eval` scores retrieval against a hand-labeled golden set:

- Dense retrieval against a BM25 keyword baseline (precision@5 and recall@10)
- Context precision over the top 10 results
- Groundedness of generated resume bullets, judged by an LLM

Build the golden set with `python -m scripts.label_golden_set --account <id>`, which runs real searches against your data and records which results are relevant. Reports are written as JSON to `evals/results/`. Pass `NO_GROUNDEDNESS=1` to skip LLM calls.

## Project layout

```
app/
  api/            FastAPI routers and page routes
  core/           settings, models, LLM client, embeddings, credential storage
  ingest/github/  GitHub client, sync, manifest parsing, cancellation
  ingest/jobs/    public URL fetch and authenticated browser fetch
  profile/        skill extraction and weighting, resume and job extraction, role families
  retrieval/      Qdrant indexing and search
  resume_build/   resume assembly, LaTeX templates, compilation, page fitting
  evals/          golden set, BM25 baseline, metrics, groundedness
  web/templates/  Jinja2 pages
scripts/          golden-set labeling and one-off migrations
tests/unit/       unit tests
tests/live/       opt-in tests against a real LLM provider
evals/            golden set and results
```

## Privacy

All data is stored locally, and embeddings are computed locally. The following text is sent to the configured LLM provider: README and description text, job posting text and screenshots, uploaded resumes, and the selected profile content used to generate a resume. Job posting text is always sent as a separate user message and is never inserted into a system prompt.

## Limitations

- Skill evidence comes from manifests, README and description text, commit statistics, and manual entries. Source files are not scanned for imports.
- Job postings are added one at a time; there is no bulk job-board crawler.
- Authenticated fetching relies on CSS selectors supplied by the user and cannot get past CAPTCHA or two-factor authentication. Automated logins may violate a site's terms of service.
- The golden set is empty in a fresh checkout, so evaluation results are only meaningful after labeling pairs against your own data.
- Editing a skill or a job posting title does not re-index it. Reprocessing the parent project, experience, or posting re-indexes it.
- Automatic resume extraction supports PDF and image files only.
- Icon glyphs in compiled PDFs do not copy as readable text.
