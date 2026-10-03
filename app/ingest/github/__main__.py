"""python -m app.ingest.github <username>: syncs one GitHub account."""

from __future__ import annotations

import logging
import sys

from app.core.db import init_db
from app.ingest.github.sync import SyncRateLimitedError, sync_account


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) < 2:
        print("usage: python -m app.ingest.github <username>")
        raise SystemExit(1)
    username = sys.argv[1]
    init_db()
    try:
        summary = sync_account(username)
    except SyncRateLimitedError as e:
        print(e)
        raise SystemExit(1) from e
    print(
        f"repos={summary.total_repos} fetched={summary.fetched} "
        f"cache_hits={summary.cache_hits}"
    )


if __name__ == "__main__":
    main()
