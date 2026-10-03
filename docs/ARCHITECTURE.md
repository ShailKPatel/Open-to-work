# Architecture

One FastAPI process serves both the JSON API and server-rendered pages. SQLite holds all application data, Qdrant holds vectors, sentence-transformers computes embeddings locally, and LiteLLM handles text and multimodal completions. Long-running work (GitHub sync, skill extraction) runs in background threads and streams progress to the browser over Server-Sent Events.

## Modules

### `app/core`

| File | Responsibility |
| --- | --- |
| `settings.py` | Fixed local paths and URLs. Only `GITHUB_TOKEN` is read from `.env`; tests and Docker Compose override the rest through the process environment. |
| `app_settings.py` | Settings picked in the app, stored in `app_settings`: the bulk and quality models (Gemini by default) and the global monthly budget. |
| `db/` | SQLAlchemy models (`models.py`), engine and `get_db()` (`engine.py`), `init_db()` and in-place migrations (`migrations.py`). |
| `llm.py` | The only module that calls an LLM provider. |
| `llm_providers.py` | Provider registry: credential fields, which fields are secret, a cheap validation request, and the extra LiteLLM arguments. |
| `api_keys_store.py` | Encrypted provider credentials, dispatch key resolution, and the recheck pass for keys that ran out of quota. |
| `key_cooldown.py` | How long an exhausted key waits before a recheck is worth making, read from the provider's own refusal (Gemini in detail, a default interval elsewhere). |
| `key_refresh.py` | Background thread that runs the due-key recheck at startup and on an interval. |
| `crypto.py` | Fernet encryption. The key comes from `APP_SECRET_KEY`, or from `data/.secret_key` (created with 0600 permissions). |
| `embeddings.py` | Local sentence-transformers embeddings (`EMBEDDING_MODEL`, fixed in code), cached by content hash and model in `embedding_cache`. `embed(model_name=)` overrides the model for callers whose vectors never meet the retrieval ones; only the skill map does. |
| `jobs.py` | In-process registry of background jobs (daemon threads with pollable state). |
| `rate_limits.py` | Append-only log of rate-limit, budget, key-failover, and run-stopped events. Writes are best-effort. |
| `pipeline.py` | Names the step a multi-step run stopped at, and the steps that had already finished, without changing the error's type. |

**Database setup.** SQLite runs in WAL mode with a 30-second busy timeout. `init_db()` does three things in order:

1. Snapshots the database file to `data/backups/` with SQLite's backup API (the 10 most recent snapshots are kept).
2. Creates any missing tables.
3. Runs idempotent column migrations: it checks `PRAGMA table_info`, then runs `ALTER TABLE ADD COLUMN` for anything missing.

To restore a snapshot, run `cp data/backups/<snapshot>.db data/open_to_work.db`.

**LLM client.** `complete(tier, messages, schema=None, account_id=None, purpose=None) -> LLMResponse` handles each call in this order:

1. Look up the cache by a hash of the model (the one picked for that tier in `app_settings`), the messages, and the schema. The tier is not part of the key, so the same model set for both tiers is one prompt, not two. Message text is normalized first (line endings, trailing spaces, blank lines), so the same README or job posting read twice is one prompt even when the whitespace moved.
2. Check the global monthly budget from `app_settings`.
3. Resolve every key for the tier's provider that may serve the account, in the order to try them.
4. For each key in turn: check that key's own budget, if it has one, then dispatch with it. A key that the provider blames (quota used up, credential rejected, model not permitted) is set aside and the next key takes over the same request. Only when every key is spent does the call fail, with one line per key saying what happened to it.
5. Mark the system message for provider-side prompt caching where the provider needs that said explicitly (Anthropic, Bedrock; Gemini and OpenAI match a repeated prefix themselves). The marking is applied to the dispatched copy only, after the cache key is computed.
6. Dispatch through LiteLLM.
7. Record an `LLMCall` row with cost, tokens, latency, `account_id`, `key_id`, and `purpose`.

`purpose` is a short label for the feature that spent the call (`repo_facts`, `resume_build`, `pagefit_trim`, ...). `/monitor`'s usage breakdown groups by it, which is what answers "what is costing money", as opposed to "which model or key was it billed through".

It raises `BudgetExceededError` before dispatch, `ApiKeyMissingError` when no usable key exists, and `LLMRateLimitedError`, which wraps LiteLLM's `RateLimitError` so callers don't need to import LiteLLM. Messages are built with `user_message(text, images=, files=)` and `system_message(text)`.

**Key failover.** A provider blaming the key is not the end of a request: `complete()` tries each stored key for that provider in turn. Errors that every key would hit the same way (the provider overloaded, a prompt too long, refused content) skip failover, so a doomed request fails once instead of once per key. `is_out_of_keys(error)` is the signal for a caller working through a batch: true means every remaining item is about to fail identically, so stop rather than spend a doomed call per item. Items never attempted keep their pending status, which is what lets the next run continue from where the last one stopped.

This matters most inside a multi-step run. A run is many separate `complete()` calls, each committing its own work, so a key dying at step three leaves steps one and two done: switching keys lets step three finish rather than stranding the run there. `app/core/pipeline.py` adds the reporting half, naming the step that stopped and the steps that had already finished, without changing the exception type the API routes map to a status.

**Keys.** Each provider can have several keys. At most one key per provider is active. A key can be disabled or restricted to a list of account ids. `resolve_dispatch_keys()` returns every enabled key that covers the account, ordered by how likely it is to answer: a key nobody has complained about, then an exhausted key whose cooldown has elapsed, then one still inside its cooldown, then one the provider rejected or blocked. Nothing is dropped, since a device with one unhappy key must still get to try it. An authentication, forbidden, or rate-limit error during dispatch updates the key's stored status, and every swap is logged to `rate_limit_events` (`key_failover`, `keys_exhausted`) so a request that survived a dying key still leaves a trail on `/monitor`.

**Exhausted keys come back on their own.** `status` splits into two groups that are handled differently. `rate_limited` is temporary: the key filled a quota window that rolls over, so the row also records `exhausted_at`, the `exhaustion_kind` (`per_minute`, `per_day`, `quota`, `unknown`) and a `retry_at` planned by `key_cooldown.py` from the provider's own refusal. Gemini's 429 body names the quota it hit and often a retry delay, so a per-minute burst waits a minute and a used-up daily allowance waits for midnight Pacific; every other provider gets a one-hour default rather than a guess at its limits. `key_refresh.py` rechecks the keys whose `retry_at` has passed, five seconds after startup and every 30 minutes after that, and `/apis` exposes the same pass (`POST /api/api-keys/recheck`, scope `due` or `exhausted`). A key that answers again loses its cooldown and goes straight back into normal rotation. `invalid` (credential rejected) and `blocked` (suspended, revoked, or the provider's API not enabled, told apart from a per-model permission error by `is_blocked_detail()`) are never rechecked automatically: waiting does not fix either, so they sit in their own section on `/apis` until someone rechecks them explicitly (scope `blocked`, or the per-key button).

The cheap check lists models rather than generating, so it proves a credential is live and cannot see how much quota is left. A recheck after the cooldown therefore means "stop treating this key as dead", and the next real call is what settles the quota; `record_dispatch_outcome()` is the only thing that learns about quota for real.

### `app/ingest/github`

- `client.py`: a PyGithub wrapper. `_call()` retries only 403/429 and 5xx responses; 404, 401, and 422 fail immediately. PyGithub's built-in retry is disabled (`retry=None`), so tenacity is the only retry layer. If a rate-limit reset is more than 10 seconds away, the client doesn't sleep through it, so a request never hangs waiting for the reset.
- `sync.py`: `sync_account`, `sync_single_repo`, and their progress generators for SSE. Repositories are upserted by `github_id`. The README, manifests, and commit stats are refetched only when `pushed_at` changes. Extraction status goes back to `pending` only when the README changed (its git blob SHA from the root listing differs from `readme_sha`, so an unchanged README isn't even downloaded) or, for a repo without one, the description changed. Any other push rewrites the manifest evidence and reweights the rest in code (`build.py`'s `refresh_repo_evidence`), with no LLM call. If GitHub rate limits a batch partway through, the generator yields `rate_limited` with the completed count instead of `done`, and the repositories already processed stay committed.
- `auto_sync.py`: a daemon thread that checks hourly for sources last synced more than 7 days ago and syncs them through `background.start_all`, then starts skill extraction. Sources that never finished a sync, are waiting on a rate-limit reset, or were attempted within the last 7 days are skipped.
- `cancellation.py`: an in-process set of cancelled run ids, checked before each repository. A run id must be unique for each sync attempt; the UI generates a UUID per click, so a leftover flag can't cancel a later sync.
- `source_parser.py`: parses a username, profile URL, or repository URL into a `ParsedSource`.
- `manifests.py`: extracts dependency names from `requirements.txt`, `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`, `Gemfile`, `pom.xml`, and `build.gradle(.kts)`.

### `app/profile`

- `manifest_skills.py`: turns manifest dependencies into skill claims deterministically, with confidence 1.0.
- `extract.py`: one bulk-tier LLM call per repository extracts both skills and project links from the README, or from the description when there is no README. If neither exists, it raises `NoSourceTextError` and the repository is marked `no_signal`. The README is cleaned before it is sent (badges, raw HTML, fenced code blocks, and boilerplate tail sections such as License and Contributing are dropped) and then cut to a character limit, so the budget is spent on prose rather than on markup that carries no skill signal. `prefetch_repo_facts()` covers several repositories per call; `build.py` runs it ahead of its per-repository loop, and anything it misses falls back to a single call, so nothing depends on the batched pass succeeding.
- `weighting.py`: computes evidence weight from evidence type, fork status, commit recency, and commit volume.
- `build.py`: `build_profile`, `build_profile_progress`, and `reprocess_repo`. Behavior:
  - Each repository commits independently.
  - A failure marks only that repository.
  - A rate limit, a budget error, or running out of usable keys stops the batch.
  - The repository's writes are committed before the LLM call, because `complete()` writes through its own session and would otherwise block on SQLite's write lock.
  - Manual evidence and manually entered links survive reprocessing.
  - Evidence is re-indexed into Qdrant on a best-effort basis.
- `jobs.py`: a background extraction worker per account. It rechecks for pending repositories after each pass, so it can run while a sync is still adding them.
- `evidence.py`: skill-evidence CRUD shared by projects and experience.
- `resume_extract.py`, `resume_ingest.py`, `resume_profile_merge.py`: take a resume upload through multimodal extraction (PDF and images) into `Skill` and `Experience` rows. Skills are deduplicated by casefolded name. Roles are matched on company and title, and dates are filled in only where they are empty. A resume this app generated is refused on upload, so the LLM's job-tailored wording never flows back into the profile: every generated PDF carries `Creator: Open to Work`, and older builds are caught by comparing bytes with the library's compiled PDFs.
- `job_extract.py`, `job_screenshot_extract.py`: extract structured fields from postings. `skills_required` is stored as a list of `{skill, level}`; `parse_skills_required()` also reads the older plain list-of-strings format.
- `skill_map.py`: builds the 2D skill map. Each skill is embedded together with its evidence context (the languages of the repositories it came from and the skills it appears beside), because bare names embed by spelling: without context, `ElasticNet` and `EfficientNet` land on top of each other. It embeds with `bge-small`, not the `EMBEDDING_MODEL` retrieval uses: the map never writes to Qdrant, and `embedding_cache` is keyed by model, so the two coexist without a reindex. Vectors are projected with t-SNE, not PCA, which preserved under a fifth of each skill's true nearest neighbours here, then clustered on the projected coordinates so the drawn groups match what is on screen. The finished layout is stored whole in `skill_map_cache` against a fingerprint of the exact texts that produced it, so `GET /api/skills/map` normally embeds nothing. `scripts/benchmark_embeddings.py` is what these choices are measured with.
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
  3. One quality-tier call returns JSON with the summary, projects, and skills, guided by length rules for the chosen template. Those rules aim slightly over the target, since the page-fit loop trims more cheaply than it fills.
  4. Any repository id or skill that wasn't a candidate is dropped.
  5. Everything left over goes into a `reserve` on the returned dict for `pagefit.py` to draw on. It never reaches a prompt or the rendered `.tex`, and the page-fit loop strips it before the content is saved.

  Job text is always a separate user message. Experience, education, and the header are rebuilt from the database and never sent to an edit call.
- `latex.py`: a Jinja2 environment with LaTeX-safe delimiters (`\BLOCK{}`, `\VAR{}`, `\#{}`), plus `escape_latex()` and `escape_latex_url()`.
- `compile.py`: runs Tectonic as a subprocess. It raises `TectonicNotInstalledError` when the binary is missing and `CompileError` when compilation fails.
- `layout.py`: the geometry the templates read (margins, section and bullet spacing, type size, leading) as parameters rather than hardcoded lengths, plus `DENSITY_LADDER`, 13 rungs from tight (9pt on `extarticle`, 0.72 cm margins) to airy (12pt, 1.5 cm). Density 1.0 reproduces each template's original geometry exactly.
- `pagefit.py`: renders at exactly the requested page count, one page or two, in both directions. It walks the density ladder for the loosest layout that still fits, which is what stops a resume from trailing off half way down its last page. Over the target after the tightest rung, it asks a multimodal model for an ordered plan of cuts (a skill, then a project bullet, then an experience bullet, never a role) and applies them one at a time, recompiling between each and only going back for a new plan once the plan runs out. One look at the PDF covers several cuts, and the PDF is the expensive part of that prompt. Under the target at the loosest rung, it adds back the account's own held-back content from the `reserve` the orchestrator hands over: candidate projects the model passed over, experience points retrieval narrowed away, unselected candidate skills. Nothing is invented, so growth costs no LLM call. `PageFitNotAchievedError` (overflow only) carries the best PDF reached; coming up short returns a `FitResult` with `fit_exact` false, since an account can legitimately not have two pages of real material.
- `background.py`: resume builds started with `POST /api/resume-build/start` run in a worker thread (`app/core/jobs.py`), so closing the tab or changing page does not stop one. Each build records its stage, its result (page fit, library resume id, finished PDF) and how much of the posting's required skills the resume lists (`match_pct`, scored like `GET /api/resume/search`). The build page polls it and picks it up again when reopened; the header reminder shows a card on every other page and a desktop notification when it ends. In memory only, like `jobs.py`: a restart loses a build in flight, and the page says so. `POST /generate` runs the same build inside the request.
- `checkpoint.py`: a background build that stops partway (model too busy, Tectonic timeout) is saved to the library as an incomplete resume. `Resume.build_state_json` keeps the original request, the tailored content with its reserve once that step has finished, and how far the page fit got (`reworded`, `cuts_made`, reported by `fit_to_page_limit`'s `on_progress`). `POST /api/resume-build/retry/{id}` carries on from there, skipping model calls that already finished, and fills in the same row on success. `checkpoint.py` turns the saved state into the done/left checklist the library shows. `content_json` stays null until the build finishes, so search and evals never see half-built content.
- `templates/`: `onepage.tex.j2` and `twopage.tex.j2`. The preamble is adapted from RenderCV (MIT). Both read their geometry from `layout.py` through a `layout` dict, so `render_resume()` can produce the same content tighter or looser.

### `app/evals`

- `golden.py`: golden-set pairs stored in YAML.
- `bm25.py`: the keyword baseline, using `rank_bm25`.
- `metrics.py`: precision@k and recall@k.
- `groundedness.py`: an LLM judge that checks generated bullets against project evidence, bounded by `max_checks`.
- `run.py`: `run_eval(account_id) -> MetricsReport` and `write_report()`. The BM25 corpus is built from SQLite using the same text builders as the dense index.

### `app/api` and `app/web`

Routers are thin. JSON endpoints live under `/api/*`, apart from `/accounts` and `/health`. Page routes return a template, and an Alpine.js component on the page loads its data from the API.

| Pages | |
| --- | --- |
| `/` | Profile picker and first-run setup |
| `/home` | Dashboard |
| `/portfolio`, `/portfolio/{projects,skills,experience,education,contact-links,resume}` | Portfolio sections and detail pages |
| `/portfolio/resume/build` | Resume generation for a posting |
| `/jobs`, `/jobs/analytics` | Job postings (one composer for pasted text and screenshots; links are kept as the apply link, never opened) and the insights dashboard |
| `/monitor`, `/monitor/sync` | Rate limits and LLM usage; GitHub sources, their syncs and skill extraction |
| `/settings`, `/apis` | Settings, API keys with models and budget |

| API prefix | Router |
| --- | --- |
| `/accounts` | `accounts.py` |
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
| `/api/api-keys` | `api_keys.py` |
| `/api/app-settings` | `app_settings.py` |
| `/api/monitor` | `monitor.py` |

Templates extend `_base.html`, which holds the theme, Tailwind (Play CDN), and Alpine.js. The shared partials are `_header.html` (top navigation) and `_portfolio_subnav.html`. The selected account id is stored in `localStorage`.

## Data model

| Table | Notes |
| --- | --- |
| `accounts` | Local profile: name, GitHub username, contact fields. Not a login. |
| `sync_sources` | GitHub users or repositories to fetch for an account. |
| `repositories` | Synced or manually added projects, with README, manifests, commit stats, extraction status, a starred flag (up to 3 per account), and an archived flag (`exclude_from_resume`). |
| `project_links` | Links per project, either `manual` or `readme_extracted`. |
| `skill_evidence` | Skill claims per repository, with evidence type, weight, and confidence. |
| `experiences`, `experience_points` | Roles and their individual bullet points. Roles can be archived. |
| `experience_skill_evidence` | Skill claims per role. |
| `education` | Degrees. Entries can be archived. |
| `skills`, `skill_stars`, `skill_archives` | Skills with no linked evidence, starred skill names, and archived skill names. |
| `social_links` | Contact links with a free-form platform. |
| `resumes` | Uploaded and generated resumes: extracted fields, `content_json`, compiled PDF path. |
| `profiles` | Aggregated skill snapshots (`skills_json`). |
| `job_postings` | Raw text (`raw_text_quarantined`), extracted fields, role family, application tracking. `content_hash` is globally unique. |
| `role_families` | Canonical job-title clusters. |
| `api_keys` | Encrypted provider credentials, masked previews, status, budget, account allow-list. |
| `app_settings` | One row per setting picked in the app: bulk model, quality model, monthly budget. A missing row means the default. |
| `llm_calls` | Cached responses plus cost, token, and latency records for every call, each tagged with the feature (`purpose`) that spent it. |
| `rate_limit_events` | Event log: rate limits, budget caps, keys swapped out or exhausted, runs stopped. |
| `embedding_cache` | Embedding vectors by content hash and model. |
| `skill_map_cache` | One stored skill-map layout per account, with the fingerprint of the skills it was built from. |

Archived projects, roles, education entries and skills stay on their pages under an Archived section but are left out of resume building, the portfolio counts, the skill map and job analytics. A skill whose every project and role is archived counts as archived too.

## Conventions

- **LLM calls.** All LLM calls go through `app.core.llm.complete()`. Nothing else imports a provider SDK.
- **Untrusted text.** Job posting text is untrusted. It is never inserted into a system prompt or formatted into a prompt template; it is sent as a separate user message that is labeled as reference material.
- **Tiers.** Bulk-tier models handle high-volume, per-item work. Quality-tier models handle single-document extraction and resume generation.
- **Source of truth.** SQLite is authoritative. Qdrant writes are best-effort: a failure is logged and never rolls back a database write.
- **Route prefixes.** A JSON route that shares a path with a page must use the `/api` prefix. FastAPI matches routes in registration order, so a collision silently hides one of them.
- **Schema changes.** A new column needs a migration function in `app/core/db/migrations.py` that follows the `PRAGMA table_info` pattern.
- **Test isolation.** Tests set `QDRANT_URL=":memory:"`, use a temporary SQLite file, and inject fakes through `_completion_fn`, `_encode_fn`, and `_run_fn`.
- **Alpine `:disabled`.** Wrap dynamic lookups in boolean attribute bindings: `:disabled="Boolean(obj[item.id])"`. Alpine 3 renders the attribute as present for a falsy property lookup.
- **Filtered inputs.** Live-filtered inputs must write the DOM value directly: `$event.target.value = form.x = sanitize($event.target.value)`. Otherwise, a keystroke that sanitizes to the unchanged model value stays visible.
