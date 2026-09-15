.PHONY: test coverage test-live lint ingest eval dev up down start

# Local (host venv): fast inner loop while writing code.
test:
	.venv/bin/pytest

# Unit tests with line and branch coverage (terminal report plus
# htmlcov/index.html). Fails if total coverage drops below fail_under in
# pyproject.toml.
coverage:
	.venv/bin/pytest --cov --cov-report=term --cov-report=html

# Opt-in suites against real services. Each file skips unless its variable
# is set, so this runs only what is configured:
#   LIVE_LLM_API_KEY (optionally LIVE_LLM_PROVIDER,  real, billed LLM calls
#     LIVE_LLM_BULK_MODEL, LIVE_LLM_QUALITY_MODEL)
#   LIVE_GITHUB=1 (optionally GITHUB_TOKEN)          real GitHub API calls
#   LIVE_APP_URL=http://localhost:8000               a running instance
# Never run by `make test` or CI. -s keeps the per-test reports.
test-live:
	.venv/bin/pytest tests/live -v -s

lint:
	.venv/bin/ruff check .
	.venv/bin/mypy app

# username is always explicit, no env default (it's a per-call input, not
# deployment config). Not named USER: that collides with the shell's own
# $USER env var, which Make inherits, so a missing argument would silently
# fall back to your OS login name instead of erroring.
# make ingest ACCOUNT=octocat
ingest:
	.venv/bin/python -m app.ingest.github $(ACCOUNT)

# make eval ACCOUNT=<account_id>: runs the retrieval/groundedness eval
# suite (app/evals/), prints a metrics table, writes the full report to
# evals/results/. Pass NO_GROUNDEDNESS=1 to skip the LLM-judge pass (no
# LLM calls, no cost, no API key needed) and only run the free
# dense-vs-BM25 retrieval comparison.
eval:
	.venv/bin/python -m app.evals $(ACCOUNT) $(if $(NO_GROUNDEDNESS),--no-groundedness,)

# Docker (one command, no host Python setup).
up:
	docker compose up --build

down:
	docker compose down

dev: up

# Friendlier front door: checks Docker, picks free ports, waits for
# health, opens the browser. `make up` still works for raw compose output.
start:
	./start.sh
