# Open to Work

[![CI](https://github.com/ShailKPatel/Open-to-work/actions/workflows/ci.yml/badge.svg)](https://github.com/ShailKPatel/Open-to-work/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Open to Work turns your GitHub into resumes built for each job. Every skill on the page is backed by real work: a dependency file, a README, a role you held. It runs on your own machine.

## Why

A resume should prove what it claims. Here the model picks from your evidence and writes around it. It cannot add a project or skill you do not have. Anything it returns outside the candidate list is thrown out.

## What it does

**Profile**

- Syncs any GitHub user or repository.
- Reads skills straight from dependency manifests (Python, JavaScript, Go, Rust, Ruby, JVM). No LLM.
- Weights evidence by recency, volume, and fork status.
- Draws a skill map from embeddings.
- Imports your existing resume from a PDF or image.

**Jobs**

- Add a posting by pasting text or dropping in screenshots.
- Pulls out salary, seniority, and required skills.
- Groups similar titles into role families.
- Shows your skill gap for each posting and skill demand across all of them.

**Resumes**

- Tailors to one posting with semantic search.
- Builds LaTeX, one page or two.
- Fits the page: tightens the layout first, then cuts one item at a time.
- Revises from a plain instruction.
- Keeps every version in a library.

## Quick start

You need Docker with Compose v2.1 or newer. Nothing else.

```bash
git clone https://github.com/ShailKPatel/Open-to-work.git open-to-work
cd open-to-work
make start
```

Then create a profile, paste an LLM API key (free Gemini keys at [Google AI Studio](https://aistudio.google.com/apikey)), sync a GitHub account, and pick a posting.

- **Linux:** `make start` installs Docker if it is missing on Ubuntu, Debian, Fedora, CentOS, and RHEL. On other distros, install Docker first.
- **macOS:** install and open [Docker Desktop](https://www.docker.com/products/docker-desktop/).
- **Windows:** install [Docker Desktop](https://www.docker.com/products/docker-desktop/), then double-click `start.cmd`. It uses port 8000. From a WSL terminal, the commands above work and pick a free port.

No `make`? Run `./start.sh`. If Docker is missing on macOS or Windows, the script opens the install page.

`make start` builds the app and Qdrant containers, waits for the health check, and opens your browser. The app takes port 8000, or the next free port. Qdrant is reachable only from the app container.

## How it works

```mermaid
flowchart LR
    GH[GitHub API] --> SYNC[ingest/github]
    SYNC --> PROF[profile<br/>extraction + weighting]
    JP[Job posting<br/>text · screenshots] --> JEX[profile<br/>job extraction + role families]
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

### Skill extraction

- Dependencies come straight from manifests. No model.
- One LLM call reads each README, or the repo description when there is no README. Repos with neither are skipped.
- Evidence is weighted by source, fork status, commit recency, and commit volume. Relevance to the job picks the skills and projects. Weight sets their order and breaks near-ties.
- A sync refetches only repos pushed since the last one. A rate limit midway keeps everything fetched so far.

### Resume generation

1. Semantic search finds candidate projects, skills, and experience bullets for the posting.
2. One LLM call picks from those candidates and writes the summary and project bullets.
3. Anything not on the candidate list is dropped. Work history and education come straight from your profile.
4. A Jinja2 LaTeX template renders the result and Tectonic compiles it.
5. If the PDF runs long, a multimodal model reads it and cuts one item at a time (a skill, then a project bullet, then an experience bullet) until it fits.

### LLM gateway

Every model call goes through one client. It caches responses by content hash, checks the monthly budget before sending, and records cost, tokens, latency, and the feature that made the call. When a key runs out, the next key for the same provider takes over and the job keeps going.

## Tech stack

| Layer | Tools |
| --- | --- |
| Backend | Python 3.12, FastAPI, SQLAlchemy, SQLite |
| Retrieval | Qdrant, bge-base-en-v1.5 (local embeddings), rank-bm25 |
| LLM | LiteLLM with Gemini, OpenAI, Anthropic, Mistral, Ollama, Azure OpenAI, AWS Bedrock |
| Frontend | Jinja2, Tailwind CSS, Alpine.js |
| Documents | Tectonic (LaTeX), pypdf |
| Quality | pytest, Ruff, mypy, GitHub Actions |
| Runtime | Docker Compose |

## Configuration

Settings live in the app. Open **Manage APIs** (`/apis`).

| Setting | Default | Purpose |
| --- | --- | --- |
| Provider keys | none | Encrypted at rest. Extra keys for a provider act as failover. |
| Bulk model | `gemini/gemini-flash-lite-latest` | Per-repo extraction, role families, groundedness checks |
| Quality model | `gemini/gemini-flash-latest` | Resume generation, page fitting, job and resume extraction |
| Monthly budget | `$20` | Hard cap across all LLM calls |

Any [LiteLLM model name](https://docs.litellm.ai/docs/providers) works, for example `ollama/llama3.1`.

One setting lives in a file, and it is optional: `GITHUB_TOKEN` in `.env` raises the GitHub API limit from 60 to 5,000 requests per hour.

All data lives in `./data`: the SQLite database, uploads, backups, and the encryption key. Each startup snapshots the database and keeps the last 10.

## Privacy

Your data and embeddings stay on your machine. Your LLM provider sees only README and description text, job posting text and screenshots, uploaded resumes, and the profile content used to build a resume. The pages load nothing from a CDN except Google Fonts. Tailwind is compiled and the JS libraries are pinned, checksum-verified and served by the app.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d qdrant
```

| Command | What it does |
| --- | --- |
| `make test` | Unit tests. No network or API key needed. |
| `make coverage` | Unit tests with line and branch coverage. Fails under 90%. |
| `make lint` | Ruff and mypy |
| `make test-live` | Checks against real services |
| `make ingest ACCOUNT=<user>` | GitHub sync from the command line |
| `make eval ACCOUNT=<id>` | Retrieval and groundedness evaluation |

### Testing

`tests/unit/` tests each layer on its own, with temporary SQLite databases, in-memory Qdrant, and fakes for GitHub, the LLM, embeddings, and Tectonic.

`tests/live/` hits real services. Each suite runs only when its variable is set:

| Variable | Checks |
| --- | --- |
| `LIVE_LLM_API_KEY` | Text, image, PDF, and multi-turn calls through the gateway (billed) |
| `LIVE_GITHUB=1` | API reachability, auth, 404 handling, single-repo sync |
| `LIVE_APP_URL` | Health, every page, and JSON endpoints on a running instance |

### Evaluation

`make eval` scores retrieval against a hand-labeled golden set:

- Dense retrieval against a BM25 baseline (precision@5, recall@10)
- Precision over the top 10 (precision@10)
- LLM-judged groundedness of generated bullets

Label your own set with `python -m scripts.label_golden_set --account <id>`. CI runs the same eval on a synthetic fixture and fails if retrieval drops below the committed baseline.

### Project layout

```
app/
  api/            FastAPI routers and page routes
  core/           settings, models, LLM client, embeddings, credential storage
  ingest/github/  GitHub client, sync, manifest parsing, cancellation
  profile/        skill extraction and weighting, resume and job extraction
  retrieval/      Qdrant indexing and search
  resume_build/   resume assembly, LaTeX templates, compilation, page fitting
  evals/          golden set, BM25 baseline, metrics, groundedness
  web/templates/  Jinja2 pages
scripts/          golden-set labeling, eval gate, embedding benchmark, maintenance
tests/            unit and live suites
```

Full design in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

[MIT](LICENSE)
