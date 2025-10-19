# test_auth_core.py
from pathlib import Path
import json
import pytest

from auth_core import (
    hash_password,
    verify_password,
    default_users,
    save_users,
    load_users,
    register_user,
    can_login,
    reset_password,
)


def test_hash_and_verify_roundtrip():
    pw = "S3cure!"
    h = hash_password(pw)
    assert h != pw
    assert verify_password(pw, h)
    assert not verify_password("wrong", h)


def test_default_users_can_login():
    users = default_users()
    assert can_login(users, "admin", "admin123")
    assert can_login(users, "jane", "pass123")
    assert not can_login(users, "admin", "wrong")
    assert not can_login(users, "ghost", "admin123")


def test_save_and_load_users(tmp_path: Path):
    users = default_users()
    p = tmp_path / "users.json"
    save_users(p, users)
    loaded = load_users(p, create_default=False)
    assert set(loaded.keys()) == set(users.keys())
    # verify that hashes work
    assert verify_password("admin123", loaded["admin"]["password_hash"])


def test_load_users_creates_default_when_missing(tmp_path: Path):
    p = tmp_path / "missing.json"
    assert not p.exists()
    users = load_users(p, create_default=True)
    assert "admin" in users and "jane" in users
    assert p.exists()


def test_load_users_returns_empty_when_missing_and_no_default(tmp_path: Path):
    p = tmp_path / "missing.json"
    users = load_users(p, create_default=False)
    assert users == {}
    assert not p.exists()


def test_load_users_recovers_from_corruption(tmp_path: Path):
    p = tmp_path / "users.json"
    p.write_text("{not json", encoding="utf-8")
    users = load_users(p, create_default=True)
    assert "admin" in users  # fell back to defaults


def test_register_user_and_login(tmp_path: Path):
    p = tmp_path / "users.json"
    users = {}
    register_user(users, "alice", "Alice A.", "alicepw")
    assert can_login(users, "alice", "alicepw")
    save_users(p, users)
    loaded = load_users(p, create_default=False)
    assert can_login(loaded, "alice", "alicepw")

    with pytest.raises(ValueError):
        register_user(users, "alice", "Another", "x")  # duplicate

    with pytest.raises(ValueError):
        register_user(users, "", "No Name", "x")  # invalid input


def test_reset_password_flow():
    users = default_users()
    # wrong current
    with pytest.raises(ValueError):
        reset_password(users, "admin", "wrong", "newpw")

    # correct flow
    reset_password(users, "admin", "admin123", "newpw")
    assert can_login(users, "admin", "newpw")
    assert not can_login(users, "admin", "admin123")


def test_password_hash_is_not_plaintext():
    users = default_users()
    admin_hash = users["admin"]["password_hash"]
    assert "admin123" not in admin_hash
