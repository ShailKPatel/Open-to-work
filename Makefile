.PHONY: test coverage test-live lint ingest eval eval-fixture eval-synthetic eval-real eval-public eval-skillspan eval-llm constraints dev up down start

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

# Retrieval eval over the invented personas and postings in
# evals/synthetic/. Seeds into a temporary database and in-process Qdrant
# and deletes both afterwards, so no Qdrant server is needed and your own
# profile is never touched. WRITE=1 saves the summary to evals/results/.
eval-synthetic:
	.venv/bin/python -m scripts.run_synthetic_eval $(if $(WRITE),--write,)

# Same, on public job postings and open-source repositories labeled by hand
# (evals/real/, see DATA_SOURCES.md there). Downloads the text into the
# gitignored evals/real/cache/ on first run; later runs reuse it.
eval-real:
	.venv/bin/python -m scripts.fetch_real_eval_data
	.venv/bin/python -m scripts.run_synthetic_eval --real $(if $(WRITE),--write,)

# Retrieval on LinkedIn software postings (evals/public/, see DATA_SOURCES.md
# there), searched with the app's own saved extraction of each posting.
# SPLIT is dev (tune against this), test (spent: the search changed after it
# was read) or fresh (the current held-out set). The extractions come from
#   make eval-llm ONLY="public public-dev public-fresh"
#   make eval-public SPLIT=fresh [WRITE=1]
eval-skillspan:
	.venv/bin/python -m scripts.fetch_skillspan
	.venv/bin/python -m scripts.run_synthetic_eval --skillspan $(if $(WRITE),--write,)

eval-public:
	.venv/bin/python -m scripts.fetch_public_eval_data
	.venv/bin/python -m scripts.run_synthetic_eval --public $(or $(SPLIT),dev) $(if $(WRITE),--write,)

# LLM-backed evals: job and resume extraction, the groundedness judge
# against human labels, and prompt injection (scripts/run_llm_evals.py).
# Real, billed calls with the key you pass, in a throwaway environment:
#   LIVE_LLM_API_KEY=... make eval-llm [WRITE=1]
# or with keys listed one per line in evals/.llm-eval-keys (gitignored).
eval-llm:
	.venv/bin/python -m scripts.fetch_real_eval_data
	.venv/bin/python -m scripts.run_llm_evals $(if $(ONLY),--only $(ONLY),) $(if $(WRITE),--write,)

# Re-pin constraints.txt to the current venv. Run deliberately, after the
# suite passes on upgraded packages, and commit with a note on what moved.
# CUDA-only packages are dropped: CI and the image use CPU torch.
constraints:
	@sed -n '/^#/p' constraints.txt > constraints.txt.new
	@.venv/bin/pip freeze --exclude-editable | grep -v -i -E '^(nvidia-|triton==|cuda-)' \
		| sort -f >> constraints.txt.new
	@mv constraints.txt.new constraints.txt
	@echo "constraints.txt updated; review the diff before committing"

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
