"""python -m app.ingest.github <username>

Username is always an explicit argument, never an env default. It's a
per-call input, not deployment config; this script isn't "your" ingest
command, it syncs whatever account you point it at.
"""

from __future__ import annotations

import logging
import sys

from app.core.db import init_db
from app.ingest.github.sync import sync_account


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) < 2:
        print("usage: python -m app.ingest.github <username>")
        raise SystemExit(1)
    username = sys.argv[1]
    init_db()
    summary = sync_account(username)
    print(
        f"repos={summary.total_repos} fetched={summary.fetched} "
        f"cache_hits={summary.cache_hits}"
    )


if __name__ == "__main__":
    main()
