# Architecture

One FastAPI process serves the JSON API and the server-rendered pages. SQLite holds the application data. Qdrant holds the vectors. sentence-transformers computes embeddings locally. LiteLLM handles text and multimodal completions. Long work (GitHub sync, skill extraction) runs in background threads and streams progress to the browser over Server-Sent Events.

## Modules

### `app/core`

| File | Job |
| --- | --- |
| `settings.py` | Fixed local paths and URLs. Only `GITHUB_TOKEN` comes from `.env`. Tests and Docker Compose set the rest through the environment. |
| `app_settings.py` | Settings chosen in the app and stored in `app_settings`: bulk model, quality model (Gemini by default), and the monthly budget. |
| `db/` | SQLAlchemy models (`models.py`), engine and `get_db()` (`engine.py`), `init_db()` and migrations (`migrations.py`). |
| `llm.py` | The one module that calls an LLM provider. |
| `llm_providers.py` | Provider registry: credential fields, which are secret, a cheap validation request, and extra LiteLLM arguments. |
| `api_keys_store.py` | Encrypted provider credentials, key selection for each call, and the recheck for keys that ran out of quota. |
| `key_cooldown.py` | How long an exhausted key waits before a recheck, read from the provider's error (Gemini in detail, a default elsewhere). |
| `key_refresh.py` | Background thread that rechecks due keys at startup and on an interval. |
| `crypto.py` | Fernet encryption. The key comes from `APP_SECRET_KEY` or from `data/.secret_key` (created with 0600 permissions). |
| `embeddings.py` | Local embeddings with `EMBEDDING_MODEL`, cached by content hash and model in `embedding_cache`. `embed(model_name=)` picks another model; only the skill map uses it. |
| `jobs.py` | In-process registry of background jobs: daemon threads with state you can poll. |
| `rate_limits.py` | Append-only log of rate limits, budget stops, key failovers, and stopped runs. |
| `pipeline.py` | Names the step where a multi-step run stopped and the steps already done, keeping the error's type. |

**Database setup.** SQLite runs in WAL mode with a 30-second busy timeout. `init_db()` does three things in order:

1. Snapshots the database to `data/backups/` with SQLite's backup API and keeps the 10 newest.
2. Creates missing tables.
3. Adds missing columns: it reads `PRAGMA table_info` and runs `ALTER TABLE ADD COLUMN` for each gap.

To restore a snapshot: `cp data/backups/<snapshot>.db data/open_to_work.db`.

**LLM client.** `complete(tier, messages, schema=None, account_id=None, purpose=None) -> LLMResponse` runs each call in this order:

1. Check the cache. The key is a hash of the model, the messages, and the schema. The tier is left out, so one model set for both tiers shares one cache entry. Whitespace in message text is normalized first, so the same README read twice hits the cache.
2. Check the monthly budget.
3. Find every key for the provider that may serve this account, in the order to try them.
4. Try each key in turn. Check its own budget, if it has one, then send. If the provider rejects the key (quota spent, bad credential, model not allowed), move to the next key with the same request. The call fails only when every key is spent, with one line per key saying why.
5. Mark the system message for provider-side prompt caching on providers that need it (Anthropic, Bedrock). Gemini and OpenAI cache a repeated prefix on their own. The mark goes on the sent copy only, after the cache key is computed.
6. Send through LiteLLM.
7. Write an `LLMCall` row with cost, tokens, latency, `account_id`, `key_id`, and `purpose`.

`purpose` names the feature that made the call (`repo_facts`, `resume_build`, `pagefit_trim`, ...). The `/monitor` usage view groups by it to show what costs money, separate from which model or key paid.

Errors: `BudgetExceededError` before sending, `ApiKeyMissingError` when no key can be used, and `LLMRateLimitedError`, which wraps LiteLLM's `RateLimitError` so callers do not import LiteLLM. Build messages with `user_message(text, images=, files=)` and `system_message(text)`.

**Key failover.** `complete()` tries each stored key for the provider in turn. Errors that every key would hit (provider overloaded, prompt too long, content refused) skip failover and fail once. In a batch, `is_out_of_keys(error)` tells the caller to stop: every remaining item would fail the same way. Items not yet tried stay pending, and the next run picks up from there.

A multi-step run is many `complete()` calls, each committing its own work. If a key dies at step three, steps one and two are already saved and the next key finishes step three. `app/core/pipeline.py` reports which step stopped and which steps finished.

**Keys.** A provider can have several keys, with one active at a time. A key can be disabled or limited to certain accounts. `resolve_dispatch_keys()` returns every enabled key for the account, best first: healthy keys, then exhausted keys past their cooldown, then keys still cooling down, then rejected or blocked keys. Every key stays in the list, so an account with one bad key still gets to try it. An auth, forbidden, or rate-limit error updates the key's status. Each swap is logged to `rate_limit_events` (`key_failover`, `keys_exhausted`) and shows on `/monitor`.

**Key recovery.** Key status falls into two groups.

- `rate_limited` is temporary. The row records `exhausted_at`, the `exhaustion_kind` (`per_minute`, `per_day`, `quota`, `unknown`), and a `retry_at` set by `key_cooldown.py` from the provider's error. Gemini's 429 names the quota it hit and often a retry delay, so a per-minute limit waits a minute and a daily limit waits for midnight Pacific. Other providers wait one hour. `key_refresh.py` rechecks keys past their `retry_at` five seconds after startup and every 30 minutes after. `/apis` runs the same check on demand (`POST /api/api-keys/recheck`, scope `due` or `exhausted`). A key that answers goes straight back into rotation.
- `invalid` (credential rejected) and `blocked` (suspended, revoked, or API not enabled; `is_blocked_detail()` tells this apart from a per-model permission error) are not rechecked on a timer. They wait in their own section on `/apis` for a manual recheck (scope `blocked`, or the button on the key).

The recheck lists models instead of generating text. It proves the credential works but cannot see remaining quota. `record_dispatch_outcome()` learns the real quota state on the next real call.

### `app/ingest/github`

- `client.py`: PyGithub wrapper. `_call()` retries 403, 429, and 5xx. 404, 401, and 422 fail at once. PyGithub's own retry is off (`retry=None`), so tenacity is the only retry layer. If a rate-limit reset is more than 10 seconds away, the client stops instead of sleeping.
- `sync.py`: `sync_account`, `sync_single_repo`, and their SSE progress generators. Repos are upserted by `github_id`. README, manifests, and commit stats are refetched only when `pushed_at` changes. Extraction goes back to `pending` only when the README changed (its blob SHA from the root listing differs from `readme_sha`, so an unchanged README is not downloaded) or, for a repo without one, the description changed. Any other push rewrites manifest evidence and reweights in code (`build.py`, `refresh_repo_evidence`) with no LLM call. A rate limit partway through yields `rate_limited` with the count done, and the finished repos stay committed.
- `auto_sync.py`: daemon thread. Every hour it syncs sources last synced more than 7 days ago through `background.start_all`, then starts skill extraction. It skips sources that never finished a sync, are waiting on a rate-limit reset, or were tried in the last 7 days.
- `cancellation.py`: in-process set of cancelled run ids, checked before each repo. The UI makes a new UUID per click, so an old flag cannot cancel a new sync.
- `source_parser.py`: parses a username, profile URL, or repo URL into a `ParsedSource`.
- `manifests.py`: reads dependency names from `requirements.txt`, `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`, `Gemfile`, `pom.xml`, and `build.gradle(.kts)`.

### `app/profile`

- `manifest_skills.py`: turns manifest dependencies into skill claims with confidence 1.0. No model.
- `extract.py`: one bulk-tier LLM call per repo pulls skills and project links from the README, or from the description when there is no README. With neither, it raises `NoSourceTextError` and marks the repo `no_signal`. The README is cleaned first (badges, raw HTML, code blocks, and tail sections like License and Contributing are dropped) and cut to a character limit, so the budget goes to prose. `prefetch_repo_facts()` handles several repos per call. `build.py` runs it before the per-repo loop, and any repo it misses gets a single call.
- `weighting.py`: evidence weight from evidence type, fork status, commit recency, and commit volume. Resume building uses it after relevance is settled: it orders the selected skills and the page-fit reserve, and breaks near-ties (within 0.02 similarity) between candidates.
- `build.py`: `build_profile`, `build_profile_progress`, and `reprocess_repo`.
  - Each repo commits on its own.
  - A failure marks only that repo.
  - A rate limit, a budget stop, or running out of keys stops the batch.
  - The repo's writes are committed before the LLM call, because `complete()` writes through its own session and would wait on SQLite's write lock.
  - Manual evidence and manual links survive reprocessing.
  - Evidence is re-indexed into Qdrant on a best-effort basis.
- `jobs.py`: one background extraction worker per account. It checks for pending repos after each pass, so it can run while a sync is still adding them.
- `evidence.py`: skill-evidence CRUD shared by projects and experience.
- `resume_extract.py`, `resume_ingest.py`, `resume_profile_merge.py`: read a resume upload (PDF or image) with a multimodal model into `Skill` and `Experience` rows. The read runs in a worker thread, so the upload returns at once. An upload left pending by a restart is marked failed at startup. Skills are deduplicated by casefolded name. Roles match on company and title, and dates fill only empty fields. Resumes made by this app are refused, so tailored wording does not flow back into the profile. Each generated PDF carries `Creator: Open to Work`, and PDFs without it are caught by comparing bytes with the library.
- `job_extract.py`, `job_screenshot_extract.py`: pull structured fields from postings. `skills_required` is a list of `{skill, level}`. `parse_skills_required()` also reads a plain list of strings.
- `injection.py`: flags posting text that looks written to steer a model (override directives, role spoofing, hidden markup, zero-width and bidi characters, output injection, encoded payloads). Detections are stored under `extracted_json["injection_flags"]` and logged, never acted on; the quarantined user-role document is the defence.
- `skill_map.py`: builds the 2D skill map. Each skill is embedded with its context (the languages of its repos and the skills next to it). Bare names embed by spelling, so `ElasticNet` and `EfficientNet` would land on top of each other. The map uses `bge-small`, not the retrieval `EMBEDDING_MODEL`. It never writes to Qdrant, and `embedding_cache` is keyed by model, so the two live side by side. Vectors are projected with t-SNE: on the committed benchmark it kept 0.530 of each skill's nearest neighbours against 0.294 for PCA (`evals/results/embeddings-20260921T111424Z.md`). Clustering runs on the projected points, so the groups match the screen. The finished layout is stored in `skill_map_cache` with a fingerprint of its input, so `GET /api/skills/map` usually embeds nothing. `scripts/benchmark_embeddings.py` runs the benchmark.
- `role_family.py`: searches the `role_families` collection (cosine similarity 0.86 or more) before creating a family with a bulk-tier LLM call. `canonical_name` is unique. If a concurrent insert wins, the existing row is used.

### `app/retrieval`

- `vectorstore.py`: Qdrant client and `ensure_collection()`. `QDRANT_URL=":memory:"` runs Qdrant in memory for tests.
- `index.py`: writers for the `skill_evidence`, `experience_points`, `resumes`, `role_families`, and `job_postings` collections. It embeds short claims (a skill with its source, a posting's extracted fields), not raw documents. Skill evidence from experience uses point id `id + 1_000_000_000` (`experience_evidence_point_id()`) so it cannot collide with repo evidence in the same collection. Deletes must use the same offset.
- `search.py`: searches each collection, filtered by account. Role families are the one global search. Skill evidence and experience points are searched hybrid: a `Query` holds the posting text and the skills it names, and the full text plus one sub-query per skill each run as a dense search (with the BGE query instruction) and as BM25 over the same points, merged by reciprocal rank fusion. `query_for_posting()` builds the query from a posting's extracted title, summary and skills, because the embedding model reads only the first 512 tokens of raw text. `mode="dense"` keeps the old single-vector search for the eval harness to report as a reference.
- `keyword.py`: BM25 over a fixed corpus (`rank_bm25`). Returns only documents sharing a token with the query, ties broken by id.

### `app/resume_build`

- `context.py`: header, experience, and education data taken straight from the database.
- `orchestrator.py`: `build_resume_data`, `build_resume_data_from_seed`, `edit_resume_content`, and `generate_resume`.
  1. Hybrid search with the posting's extracted query (`query_for_posting`) finds candidate projects and skills. The fused score sets the order. Evidence weight breaks near-ties.
  2. Experience points are picked per role, up to 5 each. If search returns nothing for a role, all its points are used.
  3. One quality-tier call returns JSON with the summary, projects, and skills, following length rules for the chosen template. The rules aim a little long, because page fit trims cheaper than it fills.
  4. Any repo id or skill that was not a candidate is dropped. Project bullets and the summary go through `grounding.py`: a bullet with a number its project's evidence does not state, or a technology the project does not have, is dropped. A project left with no bullets falls back to its description. A summary sentence with an unsupported number is removed. The selected skills are then ordered by evidence weight.
  5. Everything left over goes into a `reserve` for `pagefit.py`. The reserve never reaches a prompt or the `.tex`, and page fit strips it before saving.

  Job text goes in its own user message. Experience, education, and the header are rebuilt from the database and are not sent to an edit call.
- `grounding.py`: the number and technology checks for bullets and the summary. Any skill the account has counts as a technology, matched case-insensitively on word boundaries with a small alias map. `pagefit.py` uses it too, so a shorter rewrite cannot name a technology the original did not.
- `latex.py`: Jinja2 environment with LaTeX-safe delimiters (`\BLOCK{}`, `\VAR{}`, `\#{}`), plus `escape_latex()` and `escape_latex_url()`.
- `compile.py`: runs Tectonic as a subprocess. Raises `TectonicNotInstalledError` when the binary is missing and `CompileError` when compilation fails.
- `layout.py`: the geometry the templates read (margins, section and bullet spacing, type size, leading) as parameters. `DENSITY_LADDER` has 13 rungs, from tight (9pt on `extarticle`, 0.72 cm margins) to loose (12pt, 1.5 cm). Density 1.0 matches each template's base geometry.
- `pagefit.py`: renders at exactly the requested page count, one or two. It walks the density ladder for the loosest layout that fits, so the last page does not end half empty. Still over at the tightest rung, it asks a multimodal model for an ordered list of cuts (a skill, then a project bullet, then an experience bullet, never a whole role). It applies them one at a time, recompiling after each, and asks for a new list only when the old one runs out. One look at the PDF covers several cuts. Still short at the loosest rung, it adds back content from the `reserve`: projects the model passed over, experience points search left out, unselected candidate skills. All of it is the account's own material, and adding it costs no LLM call. `PageFitNotAchievedError` (overflow only) carries the best PDF reached. Coming up short returns a `FitResult` with `fit_exact` false.
- `background.py`: builds started with `POST /api/resume-build/start` run in a worker thread (`app/core/jobs.py`), so closing the tab does not stop them. Each build records its stage, its result (page fit, library resume id, PDF), and `match_pct`, the share of the posting's required skills the resume lists (scored like `GET /api/resume/search`). The build page polls it and picks it up again when reopened. Other pages show a header card, and the browser shows a desktop notification when the build ends. State is held in memory, like `jobs.py`. `POST /generate` runs the same build inside the request.
- `checkpoint.py`: a background build that stops partway (model busy, Tectonic timeout) is saved to the library as incomplete. `Resume.build_state_json` keeps the request, the tailored content with its reserve once that step is done, and page-fit progress (`reworded`, `cuts_made`, from `fit_to_page_limit`'s `on_progress`). `POST /api/resume-build/retry/{id}` resumes from there, skips model calls already done, and fills the same row on success. The library shows a done/left checklist built from the saved state. `content_json` stays null until the build finishes, so search and evals see only finished content.
- `templates/`: `onepage.tex.j2` and `twopage.tex.j2`. The preamble is adapted from RenderCV (MIT). Both read their geometry from `layout.py` through a `layout` dict, so `render_resume()` can set the same content tighter or looser.

### `app/evals`

- `golden.py`: golden-set pairs stored in YAML.
- `bm25.py`: the keyword baseline, re-exported from `app/retrieval/keyword.py`.
- `metrics.py`: precision@k (divided by k), recall@k, nDCG@k, reciprocal rank, percentile bootstrap intervals, and paired differences between two systems over the same pairs.
- `candidates.py`: scores what the resume builder hands the model (`_candidate_skills` by name, `_candidate_projects` by repo) against the golden pairs, so a change in candidate ranking shows up in the numbers. The CI gate checks these against `evals/ci_baseline.json` with the retrieval numbers.
- `groundedness.py`: an LLM judge that checks generated bullets against project evidence, capped by `max_checks`.
- `run.py`: `run_eval(account_id) -> MetricsReport` and `write_report()`. Scores three systems on every pair: the app's hybrid retrieval, the single-query dense search it replaced, and BM25, with paired bootstrap differences. The BM25 corpus is built from SQLite with the same text builders as the index. Pairs marked `partial` are excluded and noted.
- `synthetic.py`: the invented eval set in `evals/synthetic/` (personas, postings, judge bullets, red team postings), its golden pairs, rendered resumes, and `isolated_environment()`, a throwaway database, in-memory Qdrant and encryption key that every seeding eval runs inside.
- `real.py`: the real-text set in `evals/real/`: public postings and pinned open-source repositories, labels, and the PII scrub applied on download.
- `injection.py`: detection and false positive rates for `app/profile/injection.py` on development, held-out and ordinary postings.
- `llm_evals.py`: billed evals of job and resume extraction, the groundedness judge against human labels (accuracy, Cohen's kappa), and extraction under prompt injection. Run by `scripts/run_llm_evals.py` only.

### `app/api` and `app/web`

Routers are thin. JSON endpoints live under `/api/*`, except `/accounts` and `/health`. Page routes return a template, and an Alpine.js component on the page loads its data from the API.

| Pages | |
| --- | --- |
| `/` | Profile picker and first-run setup |
| `/home` | Dashboard |
| `/portfolio`, `/portfolio/{projects,skills,experience,education,contact-links,resume}` | Portfolio sections and detail pages |
| `/portfolio/resume/build` | Resume generation for a posting |
| `/jobs`, `/jobs/analytics` | Job postings (one form for text and screenshots; links are saved as the apply link) and the insights dashboard |
| `/monitor`, `/monitor/sync` | Rate limits and LLM usage; GitHub sources, syncs, and skill extraction |
| `/settings`, `/apis` | Settings; API keys, models, and budget |

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

Templates extend `_base.html`, which holds the theme and loads `/static/app.css` and Alpine.js. Both are built into the image by the Dockerfile's `assets` stage: Tailwind's standalone CLI compiles only the classes the templates use, and the JS libraries are pinned and checksum-verified. Nothing loads from a CDN except Google Fonts. Shared partials: `_header.html` (top navigation) and `_portfolio_subnav.html`. The selected account id is kept in `localStorage`.

## Data model

| Table | Contents |
| --- | --- |
| `accounts` | Local profile: name, GitHub username, contact fields. Not a login. |
| `sync_sources` | GitHub users or repos to fetch for an account. |
| `repositories` | Synced or manually added projects: README, manifests, commit stats, extraction status, a starred flag (up to 3 per account), and an archived flag (`exclude_from_resume`). |
| `project_links` | Links per project, `manual` or `readme_extracted`. |
| `skill_evidence` | Skill claims per repo, with evidence type, weight, and confidence. |
| `experiences`, `experience_points` | Roles and their bullet points. Roles can be archived. |
| `experience_skill_evidence` | Skill claims per role. |
| `education` | Degrees. Entries can be archived. |
| `skills`, `skill_stars`, `skill_archives` | Skills with no linked evidence, starred skill names, and archived skill names. |
| `social_links` | Contact links with a free-form platform. |
| `resumes` | Uploaded and generated resumes: extracted fields, `content_json`, compiled PDF path. |
| `job_postings` | Raw text (`raw_text_quarantined`), extracted fields, role family, application tracking. `content_hash` is unique per account: a hash of the pasted text, or of the text and screenshot bytes. |
| `role_families` | Canonical job-title clusters. |
| `api_keys` | Encrypted provider credentials, masked previews, status, budget, account allow-list. |
| `app_settings` | One row per setting chosen in the app: bulk model, quality model, monthly budget. No row means the default. |
| `llm_calls` | Cached responses with cost, tokens, and latency for every call, tagged with its `purpose`. |
| `rate_limit_events` | Event log: rate limits, budget stops, key swaps and exhaustion, stopped runs. |
| `embedding_cache` | Embedding vectors by content hash and model. |
| `skill_map_cache` | One stored skill-map layout per account, with the fingerprint of its input. |

Saving a job posting returns at once with `extraction_status` "pending". The LLM read (text extraction or screenshot transcription) runs in a worker thread through `app/core/jobs.py`, keyed by posting id, so a repeated save or a second Reprocess does not start a second read. The jobs list and the posting page poll until the status changes. Rows still pending at startup are marked failed with a note to reprocess.

Archived projects, roles, education entries, and skills stay on their pages under an Archived section. They are left out of resume building, portfolio counts, the skill map, and job analytics. A skill whose every project and role is archived counts as archived.

## Security

The app has no login. It is built for one person on their own machine, and the app's port binds to 127.0.0.1. Qdrant publishes no port at all: it has no login and no Host check, so only the app reaches it, over the compose network (`docker-compose.dev.yml` publishes it for host-side development). `app/api/security.py` stops other web pages from reaching it through that person's browser:

- **Cross-site requests.** POST, PUT, PATCH, and DELETE get a 403 when `Sec-Fetch-Site` is anything but `same-origin` or `none`, or when `Origin` (or `Referer` without `Origin`) is not the app's own origin. A request with none of these headers is not from a web page and is allowed.
- **DNS rebinding.** Every request must carry a loopback `Host` (`localhost`, `127.0.0.1`, `[::1]`, any port), or it gets a 400.
- **Response headers.** `X-Content-Type-Options: nosniff`, `Referrer-Policy: same-origin`, `X-Frame-Options: SAMEORIGIN`, and a CSP of `frame-ancestors 'self'; base-uri 'self'; form-action 'self'`. Framing allows `'self'` because the resume pages show PDFs in an iframe.
- **Uploaded files.** Screenshots must be PNG, JPEG, WebP, or GIF by their bytes. Resumes are typed by their bytes and served with that type. Only PDFs and images display inline. Everything else downloads as `application/octet-stream`, so an uploaded page cannot run on the app's origin. Stored file names are cut to their base name and prefixed with an id or hash.
- **LaTeX.** Every value in a resume template goes through `escape_latex` or `escape_latex_url` (`app/resume_build/latex.py`). Tectonic runs with shell escape off, in untrusted mode.
- **Secrets.** API keys are encrypted at rest and shown only as masked previews. Provider keys travel in request headers, not URLs.
- **Outbound requests.** Links in job postings are stored and not opened. The server calls only GitHub, the configured LLM providers, and the embedding model and Tectonic package downloads. The one user-set URL it calls is the `api_base` of an Azure OpenAI or Ollama key.
- **Build inputs.** Every file the Dockerfile downloads (Tailwind CLI, Tectonic, Alpine.js, marked, DOMPurify) is pinned to a version and checked against its SHA-256 with `ADD --checksum`. The Qdrant image is pinned to a version.
- **Untrusted text.** Job posting text and repo README and description text go to the model as separate user messages labeled as reference material. They are not placed in a system prompt or formatted into a prompt template.
