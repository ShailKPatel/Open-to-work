.PHONY: test coverage test-live lint ingest eval eval-fixture dev up down start

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

# The eval gate CI runs on every pull request, reproduced locally so a red
# gate can be debugged without pushing. Seeds the synthetic fixture account
# (scripts/seed_eval_fixture.py), scores it against the committed golden
# pairs, and compares the result to evals/ci_baseline.json. Also the command
# to run when a change legitimately moves the numbers and the baseline needs
# updating.
#
# Needs a Qdrant server: `make up`, or set QDRANT_URL at any running one.
# Everything it writes goes to .eval-fixture/ (gitignored), so neither your
# real database nor evals/results/ is touched.
EVAL_FIXTURE_DIR := .eval-fixture
EVAL_FIXTURE_ENV := DATABASE_URL="sqlite:///$(CURDIR)/$(EVAL_FIXTURE_DIR)/fixture.db" \
	EVALS_GOLDEN_DIR="$(EVAL_FIXTURE_DIR)/golden" \
	EVALS_RESULTS_DIR="$(EVAL_FIXTURE_DIR)/results"

eval-fixture:
	@mkdir -p $(EVAL_FIXTURE_DIR)/golden $(EVAL_FIXTURE_DIR)/results
	@cp evals/golden/ci_fixture.yaml $(EVAL_FIXTURE_DIR)/golden/golden_set.yaml
	$(EVAL_FIXTURE_ENV) .venv/bin/python -m scripts.seed_eval_fixture
	@REPORT=$$($(EVAL_FIXTURE_ENV) .venv/bin/python -m app.evals 9001 --no-groundedness \
		| tee /dev/stderr | sed -n 's/^Written to //p'); \
	.venv/bin/python -m scripts.check_eval_baseline \
		--report "$$REPORT" --baseline evals/ci_baseline.json

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
