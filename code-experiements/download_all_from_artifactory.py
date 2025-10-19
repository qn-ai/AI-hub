# auth_core.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import bcrypt


@dataclass
class UserRecord:
    name: str
    password_hash: str


def hash_password(password: str) -> str:
    """Hash a password with bcrypt and return the utf-8 string hash."""
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    if not isinstance(password, str) or not isinstance(password_hash, str):
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def default_users() -> Dict[str, Dict[str, str]]:
    """Return a default demo user set (admin/admin123, jane/pass123)."""
    return {
        "admin": {"name": "Admin User", "password_hash": hash_password("admin123")},
        "jane": {"name": "Jane Doe", "password_hash": hash_password("pass123")},
    }


def save_users(path: Path, users: Dict[str, Dict[str, str]]) -> None:
    """Save users dictionary to JSON at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def load_users(path: Path, create_default: bool = True) -> Dict[str, Dict[str, str]]:
    """Load users from JSON. If not exists or invalid and create_default, return and write defaults."""
    if not path.exists():
        if create_default:
            users = default_users()
            save_users(path, users)
            return users
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Users JSON must be an object")
            return data
    except Exception:
        if create_default:
            users = default_users()
            save_users(path, users)
            return users
        return {}


def register_user(users: Dict[str, Dict[str, str]], username: str, full_name: str, password: str) -> None:
    """Add a new user to the users dict. Raises ValueError if invalid or exists."""
    if not username or not full_name or not password:
        raise ValueError("All fields are required")
    if username in users:
        raise ValueError("Username already exists")
    users[username] = {"name": full_name, "password_hash": hash_password(password)}


def can_login(users: Dict[str, Dict[str, str]], username: str, password: str) -> bool:
    """Return True if username exists and password verifies."""
    rec = users.get(username)
    if not rec or "password_hash" not in rec:
        return False
    return verify_password(password, rec["password_hash"])


def reset_password(users: Dict[str, Dict[str, str]], username: str, old_password: str, new_password: str) -> None:
    """Reset password for a user after verifying old password. Raises ValueError on failure."""
    if username not in users:
        raise ValueError("User not found")
    if not verify_password(old_password, users[username]["password_hash"]):
        raise ValueError("Current password is incorrect")
    if not new_password:
        raise ValueError("New password cannot be empty")
    users[username]["password_hash"] = hash_password(new_password)
