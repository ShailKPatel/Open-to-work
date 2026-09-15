# Architecture

One FastAPI process serves both the JSON API and server-rendered pages. SQLite holds all application data, Qdrant holds vectors, sentence-transformers computes embeddings locally, and LiteLLM handles text and multimodal completions. Long-running work (GitHub sync, skill extraction) runs in background threads and streams progress to the browser over Server-Sent Events.

## Modules

### `app/core`

| File | Responsibility |
| --- | --- |
| `settings.py` | Fixed local paths and URLs. Only `GITHUB_TOKEN` is read from `.env`; tests and Docker Compose override the rest through the process environment. |
| `app_settings.py` | Settings picked in the app, stored in `app_settings`: the bulk and quality models (Gemini by default) and the global monthly budget. |
| `db.py` | SQLAlchemy models, engine, `get_db()`, `init_db()`. |
| `llm.py` | The only module that calls an LLM provider. |
| `llm_providers.py` | Provider registry: credential fields, which fields are secret, a cheap validation request, and the extra LiteLLM arguments. |
| `api_keys_store.py` | Encrypted provider credentials and dispatch key resolution. |
| `auth_sources_store.py` | Encrypted login profiles for authenticated job fetching. |
| `crypto.py` | Fernet encryption. The key comes from `APP_SECRET_KEY`, or from `data/.secret_key` (created with 0600 permissions). |
| `embeddings.py` | Local sentence-transformers embeddings (`EMBEDDING_MODEL`, fixed in code), cached by content hash in `embedding_cache`. |
| `jobs.py` | In-process registry of background jobs (daemon threads with pollable state). |
| `rate_limits.py` | Append-only log of rate-limit and budget events. Writes are best-effort. |

**Database setup.** SQLite runs in WAL mode with a 30-second busy timeout. `init_db()` does three things in order:

1. Snapshots the database file to `data/backups/` with SQLite's backup API (the 10 most recent snapshots are kept).
2. Creates any missing tables.
3. Runs idempotent column migrations: it checks `PRAGMA table_info`, then runs `ALTER TABLE ADD COLUMN` for anything missing.

To restore a snapshot, run `cp data/backups/<snapshot>.db data/open_to_work.db`.

**LLM client.** `complete(tier, messages, schema=None, account_id=None) -> LLMResponse` handles each call in this order:

1. Look up the cache by a hash of tier, model (the one picked for that tier in `app_settings`), messages, and schema.
2. Check the global monthly budget from `app_settings`.
3. Resolve a key for the tier's provider and the account.
4. Check that key's own budget, if it has one.
5. Dispatch through LiteLLM.
6. Record an `LLMCall` row with cost, tokens, latency, `account_id`, and `key_id`.

It raises `BudgetExceededError` before dispatch, `ApiKeyMissingError` when no usable key exists, and `LLMRateLimitedError`, which wraps LiteLLM's `RateLimitError` so callers don't need to import LiteLLM. Messages are built with `user_message(text, images=, files=)` and `system_message(text)`.

**Keys.** Each provider can have several keys. At most one key per provider is active. A key can be disabled or restricted to a list of account ids. `resolve_dispatch_key()` prefers the active key and otherwise falls back to the next enabled key that covers the account. An authentication or rate-limit error during dispatch updates the key's stored status.

### `app/ingest/github`

- `client.py`: a PyGithub wrapper. `_call()` retries only 403/429 and 5xx responses; 404, 401, and 422 fail immediately. PyGithub's built-in retry is disabled (`retry=None`), so tenacity is the only retry layer. If a rate-limit reset is more than 10 seconds away, the client doesn't sleep through it, so a request never hangs waiting for the reset.
- `sync.py`: `sync_account`, `sync_single_repo`, and their progress generators for SSE. Repositories are upserted by `github_id`. The README, manifests, and commit stats are refetched only when `pushed_at` changes, which also resets extraction status to `pending`. If GitHub rate limits a batch partway through, the generator yields `rate_limited` with the completed count instead of `done`, and the repositories already processed stay committed.
- `cancellation.py`: an in-process set of cancelled run ids, checked before each repository. A run id must be unique for each sync attempt; the UI generates a UUID per click, so a leftover flag can't cancel a later sync.
- `source_parser.py`: parses a username, profile URL, or repository URL into a `ParsedSource`.
- `manifests.py`: extracts dependency names from `requirements.txt`, `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`, `Gemfile`, `pom.xml`, and `build.gradle(.kts)`.

### `app/ingest/jobs`

- `url_fetch.py`: fetches with httpx and extracts text with BeautifulSoup. A page with less than 200 characters of text after markup is stripped is rejected, since that usually means a login wall or a client-rendered page. Text is capped at 20,000 characters.
- `auth_fetch.py`: uses headless Chromium through Playwright. Each fetch opens a fresh browser context, logs in with the stored selectors, reads one page, and closes the browser. No cookies or sessions are kept.

### `app/profile`

- `manifest_skills.py`: turns manifest dependencies into skill claims deterministically, with confidence 1.0.
- `extract.py`: makes bulk-tier LLM calls that extract skills and project links from the README, or from the description when there is no README. If neither exists, it raises `NoSourceTextError` and the repository is marked `no_signal`.
- `weighting.py`: computes evidence weight from evidence type, fork status, commit recency, and commit volume.
- `build.py`: `build_profile`, `build_profile_progress`, and `reprocess_repo`. Behavior:
  - Each repository commits independently.
  - A failure marks only that repository.
  - A rate limit or budget error stops the batch.
  - The repository's writes are committed before the LLM call, because `complete()` writes through its own session and would otherwise block on SQLite's write lock.
  - Manual evidence and manually entered links survive reprocessing.
  - Evidence is re-indexed into Qdrant on a best-effort basis.
- `jobs.py`: a background extraction worker per account. It rechecks for pending repositories after each pass, so it can run while a sync is still adding them.
- `evidence.py`: skill-evidence CRUD shared by projects and experience.
- `resume_extract.py`, `resume_ingest.py`, `resume_profile_merge.py`: take a resume upload through multimodal extraction (PDF and images) into `Skill` and `Experience` rows. Skills are deduplicated by casefolded name. Roles are matched on company and title, and dates are filled in only where they are empty.
- `job_extract.py`, `job_screenshot_extract.py`: extract structured fields from postings. `skills_required` is stored as a list of `{skill, level}`; `parse_skills_required()` also reads the older plain list-of-strings format.
- `role_family.py`: searches the `role_families` collection (cosine similarity of at least 0.86) before creating a new family with a bulk-tier LLM call. `canonical_name` is unique; if a concurrent insert wins, the existing row is reused.

### `app/retrieval`

- `vectorstore.py`: the Qdrant client and `ensure_collection()`. Setting `QDRANT_URL=":memory:"` runs Qdrant in memory for tests.
- `index.py`: writers for the `skill_evidence`, `experience_points`, `resumes`, `role_families`, and `job_postings` collections. It embeds reduced claims (a skill with its source, or a posting's extracted fields) rather than raw documents. Experience-linked skill evidence uses point id `id + 1_000_000_000` (`experience_evidence_point_id()`) so it can't collide with repository evidence in the shared collection; deletes must apply the same offset.
- `search.py`: searches each collection, filtered by account. Role families are the only global search.

### `app/resume_build`

- `context.py`: deterministic header, experience, and education data taken straight from the database.
- `orchestrator.py`: `build_resume_data`, `build_resume_data_from_seed`, `edit_resume_content`, and `generate_resume`. Steps:
  1. Candidate projects and skills come from semantic search over the posting text.
  2. Experience points are selected per role (up to 5 per role, falling back to all of the role's points if the search returns nothing).
  3. One quality-tier call returns JSON with the summary, projects, and skills, guided by length rules for the chosen template.
  4. Any repository id or skill that wasn't a candidate is dropped.

  Job text is always a separate user message. Experience, education, and the header are rebuilt from the database and never sent to an edit call.
- `latex.py`: a Jinja2 environment with LaTeX-safe delimiters (`\BLOCK{}`, `\VAR{}`, `\#{}`), plus `escape_latex()` and `escape_latex_url()`.
- `compile.py`: runs Tectonic as a subprocess. It raises `TectonicNotInstalledError` when the binary is missing and `CompileError` when compilation fails.
- `pagefit.py`: compiles, counts pages with pypdf, and asks a multimodal model for one cut at a time until the resume fits. `PageFitNotAchievedError` carries the best PDF reached.
- `templates/`: `onepage.tex.j2` and `twopage.tex.j2`. The preamble is adapted from RenderCV (MIT).

### `app/evals`

- `golden.py`: golden-set pairs stored in YAML.
- `bm25.py`: the keyword baseline, using `rank_bm25`.
- `metrics.py`: precision@k and recall@k.
- `groundedness.py`: an LLM judge that checks generated bullets against project evidence, bounded by `max_checks`.
- `run.py`: `run_eval(account_id) -> MetricsReport` and `write_report()`. The BM25 corpus is built from SQLite using the same text builders as the dense index.

### `app/api` and `app/web`

Routers are thin. JSON endpoints live under `/api/*`, apart from `/accounts`, `/sync/github`, and `/health`. Page routes return a template, and an Alpine.js component on the page loads its data from the API.

| Pages | |
| --- | --- |
| `/` | Profile picker and first-run setup |
| `/home` | Dashboard |
| `/sync` | Initial GitHub sync |
| `/portfolio`, `/portfolio/{projects,skills,experience,education,contact,resume}` | Portfolio sections and detail pages |
| `/portfolio/resume/build` | Resume generation for a posting |
| `/jobs`, `/jobs/analytics` | Job postings and skill-demand analytics |
| `/monitor` | Rate limits and LLM usage |
| `/settings`, `/settings/sources`, `/settings/auth-sources`, `/apis` | Settings, GitHub sources, login profiles, API keys with models and budget |

| API prefix | Router |
| --- | --- |
| `/accounts` | `accounts.py` |
| `/sync/github` | `main.py` (sync, SSE stream, cancel) |
| `/api/projects` | `projects.py` |
| `/api/skills` | `skills.py` |
| `/api/experience` | `experience.py` |
| `/api/education` | `education.py` |
| `/api/accounts/{id}/contact`, `/api/accounts/{id}/social-links` | `contact.py` |
| `/api/sources` | `sources.py` |
| `/api/resume` | `resume.py` |
| `/api/resume-build` | `resume_build.py` |
| `/api/job-postings` | `job_postings.py` |
| `/api/job-analytics` | `job_analytics.py` |
| `/api/auth-sources` | `auth_sources.py` |
| `/api/api-keys` | `api_keys.py` |
| `/api/app-settings` | `app_settings.py` |
| `/api/monitor` | `monitor.py` |

Templates extend `_base.html`, which holds the theme, Tailwind (Play CDN), and Alpine.js. The shared partials are `_header.html` (top navigation) and `_portfolio_subnav.html`. The selected account id is stored in `localStorage`.

## Data model

| Table | Notes |
| --- | --- |
| `accounts` | Local profile: name, GitHub username, contact fields. Not a login. |
| `sync_sources` | GitHub users or repositories to fetch for an account. |
| `repositories` | Synced or manually added projects, with README, manifests, commit stats, extraction status, and a starred flag (up to 3 per account). |
| `project_links` | Links per project, either `manual` or `readme_extracted`. |
| `skill_evidence` | Skill claims per repository, with evidence type, weight, and confidence. |
| `experiences`, `experience_points` | Roles and their individual bullet points. |
| `experience_skill_evidence` | Skill claims per role. |
| `education` | Degrees. |
| `skills`, `skill_stars` | Skills with no linked evidence, and starred skill names. |
| `social_links` | Contact links with a free-form platform. |
| `resumes` | Uploaded and generated resumes: extracted fields, `content_json`, compiled PDF path. |
| `profiles` | Aggregated skill snapshots (`skills_json`). |
| `job_postings` | Raw text (`raw_text_quarantined`), extracted fields, role family, application tracking. `content_hash` is globally unique. |
| `role_families` | Canonical job-title clusters. |
| `auth_sources` | Encrypted login profiles and CSS selectors. |
| `api_keys` | Encrypted provider credentials, masked previews, status, budget, account allow-list. |
| `app_settings` | One row per setting picked in the app: bulk model, quality model, monthly budget. A missing row means the default. |
| `llm_calls` | Cached responses plus cost, token, and latency records for every call. |
| `rate_limit_events` | Rate-limit and budget event log. |
| `embedding_cache` | Embedding vectors by content hash and model. |
| `detections`, `match_results` | Reserved; not yet written to. |

## Conventions

- **LLM calls.** All LLM calls go through `app.core.llm.complete()`. Nothing else imports a provider SDK.
- **Untrusted text.** Job posting text is untrusted. It is never inserted into a system prompt or formatted into a prompt template; it is sent as a separate user message that is labeled as reference material.
- **Tiers.** Bulk-tier models handle high-volume, per-item work. Quality-tier models handle single-document extraction and resume generation.
- **Source of truth.** SQLite is authoritative. Qdrant writes are best-effort: a failure is logged and never rolls back a database write.
- **Route prefixes.** A JSON route that shares a path with a page must use the `/api` prefix. FastAPI matches routes in registration order, so a collision silently hides one of them.
- **Schema changes.** A new column needs a migration function in `db.py` that follows the `PRAGMA table_info` pattern.
- **Test isolation.** Tests set `QDRANT_URL=":memory:"`, use a temporary SQLite file, and inject fakes through `_completion_fn`, `_encode_fn`, `_run_fn`, and `_playwright_fn`.
- **Alpine `:disabled`.** Wrap dynamic lookups in boolean attribute bindings: `:disabled="Boolean(obj[item.id])"`. Alpine 3 renders the attribute as present for a falsy property lookup.
- **Filtered inputs.** Live-filtered inputs must write the DOM value directly: `$event.target.value = form.x = sanitize($event.target.value)`. Otherwise, a keystroke that sanitizes to the unchanged model value stays visible.
