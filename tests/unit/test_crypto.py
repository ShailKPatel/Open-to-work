"""Encryption at rest for stored secrets (app/core/crypto.py): round trips,
key source precedence, key file permissions, and failure modes."""

import os
import stat

import pytest
from cryptography.fernet import Fernet

import app.core.crypto as crypto


@pytest.fixture(autouse=True)
def _isolated_key(tmp_path, monkeypatch):
    """Every test gets its own key file location and a fresh Fernet cache,
    so no test reads or writes the real data/.secret_key."""
    monkeypatch.delenv("APP_SECRET_KEY", raising=False)
    monkeypatch.setattr(crypto, "_SECRET_KEY_PATH", tmp_path / "data" / ".secret_key")
    monkeypatch.setattr(crypto, "_fernet", None)


def test_encrypt_then_decrypt_round_trips():
    token = crypto.encrypt("sk-live-123")
    assert token != "sk-live-123"
    assert "sk-live-123" not in token
    assert crypto.decrypt(token) == "sk-live-123"


def test_unicode_round_trips():
    assert crypto.decrypt(crypto.encrypt("pässwörd ✓")) == "pässwörd ✓"


def test_same_plaintext_encrypts_differently_each_time():
    assert crypto.encrypt("same") != crypto.encrypt("same")


def test_env_key_takes_precedence_and_no_key_file_is_written(monkeypatch):
    key = Fernet.generate_key()
    monkeypatch.setenv("APP_SECRET_KEY", key.decode())

    token = crypto.encrypt("secret")

    assert Fernet(key).decrypt(token.encode()) == b"secret"
    assert not crypto._SECRET_KEY_PATH.exists()


def test_generated_key_file_is_owner_read_write_only():
    crypto.encrypt("x")

    path = crypto._SECRET_KEY_PATH
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_existing_key_file_is_reused_after_restart(monkeypatch):
    token = crypto.encrypt("persisted")
    monkeypatch.setattr(crypto, "_fernet", None)  # new process, same data dir

    assert crypto.decrypt(token) == "persisted"


def test_lost_creation_race_uses_the_other_process_key(monkeypatch):
    """Another process creates the key file between the exists() check and
    the exclusive create. The loser must read that key, not crash."""
    path = crypto._SECRET_KEY_PATH
    theirs = Fernet.generate_key()
    real_open = os.open

    def racing_open(target, flags, mode=0o777):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(theirs)
        return real_open(target, flags, mode)

    monkeypatch.setattr(crypto.os, "open", racing_open)

    assert crypto._load_or_create_key() == theirs


def test_decrypt_with_a_different_key_raises_value_error(monkeypatch):
    token = crypto.encrypt("x")
    monkeypatch.setattr(crypto, "_fernet", None)
    monkeypatch.setenv("APP_SECRET_KEY", Fernet.generate_key().decode())

    with pytest.raises(ValueError, match="could not decrypt"):
        crypto.decrypt(token)


def test_decrypt_garbage_raises_value_error():
    with pytest.raises(ValueError, match="could not decrypt"):
        crypto.decrypt("not-a-fernet-token")
