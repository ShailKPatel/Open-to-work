"""Downloads the SkillSpan dataset's three splits (CC BY 4.0) into
evals/skillspan/cache/ (gitignored). See app/evals/skillspan.py for how
they are used.

Usage: .venv/bin/python -m scripts.fetch_skillspan
"""

from __future__ import annotations

import httpx

from app.evals.skillspan import CACHE_DIR, DATASET_URL, SPLITS


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = CACHE_DIR / f"{split}.json"
        if path.exists():
            print(f"{path} already cached")
            continue
        response = httpx.get(DATASET_URL.format(split=split), follow_redirects=True, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
        print(f"cached {path}")


if __name__ == "__main__":
    main()
