"""Encryption at rest for secrets we store in the DB (every provider's
credentials in app/core/db/models.py's ApiKey table, app/core/api_keys_store.py).
Nothing upstream of this module should ever write a raw secret straight
into a DB column.

The Fernet key lives outside the database: encrypting a secret with a key
stored in the same file would not protect anything. `APP_SECRET_KEY` in the
environment wins if set (so a deployment can pin or rotate it); otherwise one is
generated once and persisted to `data/.secret_key` with owner-only (0600)
file permissions, so another local user/process can't just read it off
disk. That file is inside `data/`, which is already gitignored.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

_SECRET_KEY_PATH = Path("./data/.secret_key")

_fernet: Fernet | None = None


def _load_or_create_key() -> bytes:
    env_key = os.environ.get("APP_SECRET_KEY")
    if env_key:
        return env_key.encode("ascii")

    if _SECRET_KEY_PATH.exists():
        return _SECRET_KEY_PATH.read_bytes().strip()

    _SECRET_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    # Create with 0600 from the start. Write-then-chmod would leave a window
    # where the file is world-readable.
    try:
        fd = os.open(
            _SECRET_KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR
        )
    except FileExistsError:
        # Lost a race with another process creating it first; read theirs.
        return _SECRET_KEY_PATH.read_bytes().strip()
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    return key


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        _fernet = Fernet(_load_or_create_key())
    return _fernet


def encrypt(plaintext: str) -> str:
    return _get_fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt(token: str) -> str:
    try:
        return _get_fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as e:
        raise ValueError("could not decrypt stored secret (key file may have changed)") from e
