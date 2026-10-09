"""Dev and test splits for the eval sets.

Nothing here is trained, so there is no training split. What needs
keeping apart is the data a prompt, rule or threshold was changed against
(dev) from the data its result is reported on (test). Changing anything
after looking at test results turns test into dev; a new test set is then
needed, never a re-run of the old one.

Items are assigned by a hash of a group key, not drawn at random, so the
split is the same on every machine and never moves when items are added.
The group key is the posting: every bullet from one generated resume
lands on the same side, so near-duplicate bullets cannot sit on both.
"""

from __future__ import annotations

import hashlib

DEV = "dev"
TEST = "test"

# Share of groups that go to test. Dev only needs enough items to see
# failure modes; test needs enough for a usable confidence interval.
TEST_SHARE = 0.7


def split_for(group: str, test_share: float = TEST_SHARE) -> str:
    """DEV or TEST for a group key, stable across runs and machines."""
    digest = hashlib.sha256(group.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return TEST if fraction < test_share else DEV
