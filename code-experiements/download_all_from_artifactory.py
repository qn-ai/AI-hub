# app_custom.py
import json
import os
from pathlib import Path
from typing import Dict, Any

import bcrypt
import streamlit as st

st.set_page_config(page_title="Custom Streamlit Login", page_icon="🔒")

USERS_PATH = Path(os.getenv("USERS_JSON_PATH", "users.json"))

# ---------------- Utilities ----------------
def _default_users() -> Dict[str, Any]:
    # Demo users: admin/admin123, jane/pass123
    return {
        "admin": {
            "name": "Admin User",
            "password_hash": bcrypt.hashpw(b"admin123", bcrypt.gensalt()).decode("utf-8"),
        },
        "jane": {
            "name": "Jane Doe",
            "password_hash": bcrypt.hashpw(b"pass123", bcrypt.gensalt()).decode("utf-8"),
        },
    }

def load_users() -> Dict[str, Any]:
    if not USERS_PATH.exists():
        users = _default_users()
        save_users(users)
        return users
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Fallback to default if file is corrupted
        users = _default_users()
        save_users(users)
        return users

def save_users(users: Dict[str, Any]) -> None:
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False

# ---------------- State helpers ----------------
if "auth" not in st.session_state:
    st.session_state["auth"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "name" not in st.session_state:
    st.session_state["name"] = None
if "login_attempts" not in st.session_state:
    st.session_state["login_attempts"] = 0

users = load_users()

# ---------------- UI ----------------
st.title("🔒 Custom Login (Streamlit + bcrypt)")

tab_login, tab_register, tab_reset = st.tabs(["Login", "Register", "Reset Password"])

with tab_login:
    st.subheader("Login")
    if st.session_state["auth"]:
        st.success(f"Already logged in as {st.session_state['name']} ({st.session_state['username']}).")
        if st.button("Logout"):
            st.session_state["auth"] = False
            st.session_state["username"] = None
            st.session_state["name"] = None
            st.info("Logged out.")
    else:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            st.session_state["login_attempts"] += 1
            if st.session_state["login_attempts"] > 10:
                st.error("Too many attempts. Please reload the page and try again.")
            else:
                if username in users and verify_password(password, users[username]["password_hash"]):
                    st.session_state["auth"] = True
                    st.session_state["username"] = username
                    st.session_state["name"] = users[username]["name"]
                    st.session_state["login_attempts"] = 0
                    st.success(f"Welcome, {st.session_state['name']}!")
                else:
                    st.error("Invalid username or password.")

with tab_register:
    st.subheader("Register")
    st.caption("Demo registration stored in a local JSON file. Do not use this as-is for production.")
    with st.form("register_form", clear_on_submit=False):
        full_name = st.text_input("Full name")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account")
    if submitted:
        if not full_name or not new_username or not new_password:
            st.warning("Please fill in all fields.")
        elif new_password != confirm_password:
            st.warning("Passwords do not match.")
        elif new_username in users:
            st.error("Username already exists.")
        else:
            users[new_username] = {"name": full_name, "password_hash": hash_password(new_password)}
            save_users(users)
            st.success(f"User '{new_username}' registered. You can log in now.")

with tab_reset:
    st.subheader("Reset Password")
    st.caption("Requires your current password. This updates the local JSON file.")
    with st.form("reset_form"):
        rp_username = st.text_input("Username", value=st.session_state.get("username") or "")
        old_password = st.text_input("Current password", type="password")
        new_password = st.text_input("New password", type="password")
        confirm_new = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Change password")
    if submitted:
        if rp_username not in users:
            st.error("User not found.")
        elif not verify_password(old_password, users[rp_username]["password_hash"]):
            st.error("Current password is incorrect.")
        elif not new_password or new_password != confirm_new:
            st.error("New passwords do not match or are empty.")
        else:
            users[rp_username]["password_hash"] = hash_password(new_password)
            save_users(users)
            st.success("Password updated successfully.")

st.divider()
if st.session_state["auth"]:
    st.header("Protected content")
    st.write("🎉 You are authenticated. Put your app here.")
    with st.sidebar:
        st.caption(f"Logged in as **{st.session_state['name']}** ({st.session_state['username']})")
        if st.button("Logout", key="logout_sidebar"):
            st.session_state["auth"] = False
            st.session_state["username"] = None
            st.session_state["name"] = None
            st.info("Logged out.")
else:
    st.info("Log in to view protected content.")

with st.sidebar.expander("Security notes"):
    st.markdown(
        """
- This demo stores users in a local **JSON** file (`users.json`). For real apps, use a database.
- Always store **hashed** passwords (bcrypt or Argon2), never plain text.
- Serve your app behind HTTPS/reverse proxy in production.
- You can set the `USERS_JSON_PATH` environment variable to control the user store location.
- Session persistence here uses `st.session_state` only; closing the tab resets it.
        """
    )
