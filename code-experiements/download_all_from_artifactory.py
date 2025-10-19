# app_custom_refactored.py
import os
from pathlib import Path

import streamlit as st

from auth_core import (
    load_users,
    save_users,
    register_user,
    can_login,
    reset_password,
)

st.set_page_config(page_title="Custom Streamlit Login (Refactored)", page_icon="🔒")

USERS_PATH = Path(os.getenv("USERS_JSON_PATH", "users.json"))
users = load_users(USERS_PATH)

# ---------------- UI ----------------
st.title("🔒 Custom Login (Refactored for Testing)")

tab_login, tab_register, tab_reset = st.tabs(["Login", "Register", "Reset Password"])

# Session state init
for key, val in {"auth": False, "username": None, "name": None, "login_attempts": 0}.items():
    st.session_state.setdefault(key, val)

with tab_login:
    st.subheader("Login")
    if st.session_state["auth"]:
        st.success(f"Logged in as {st.session_state['name']} ({st.session_state['username']}).")
        if st.button("Logout"):
            for k in ["auth", "username", "name"]:
                st.session_state[k] = None if k != "auth" else False
            st.info("Logged out.")
    else:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            st.session_state["login_attempts"] += 1
            if st.session_state["login_attempts"] > 10:
                st.error("Too many attempts. Please reload.")
            elif can_login(users, username, password):
                st.session_state["auth"] = True
                st.session_state["username"] = username
                st.session_state["name"] = users[username]["name"]
                st.session_state["login_attempts"] = 0
                st.success(f"Welcome, {st.session_state['name']}!")
            else:
                st.error("Invalid username or password.")

with tab_register:
    st.subheader("Register")
    with st.form("register_form"):
        full_name = st.text_input("Full name")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account")
    if submitted:
        if new_password != confirm_password:
            st.warning("Passwords do not match.")
        else:
            try:
                register_user(users, new_username, full_name, new_password)
                save_users(USERS_PATH, users)
                st.success(f"User '{new_username}' registered. You can log in now.")
            except Exception as e:
                st.error(str(e))

with tab_reset:
    st.subheader("Reset Password")
    with st.form("reset_form"):
        rp_username = st.text_input("Username", value=st.session_state.get("username") or "")
        old_password = st.text_input("Current password", type="password")
        new_password = st.text_input("New password", type="password")
        confirm_new = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Change password")
    if submitted:
        if new_password != confirm_new:
            st.error("New passwords do not match.")
        else:
            try:
                reset_password(users, rp_username, old_password, new_password)
                save_users(USERS_PATH, users)
                st.success("Password updated successfully.")
            except Exception as e:
                st.error(str(e))

st.divider()
if st.session_state["auth"]:
    st.header("Protected content")
    st.write("🎉 You are authenticated. Put your app here.")
    with st.sidebar:
        st.caption(f"Logged in as **{st.session_state['name']}** ({st.session_state['username']})")
        if st.button("Logout", key="logout_sidebar"):
            for k in ["auth", "username", "name"]:
                st.session_state[k] = None if k != "auth" else False
            st.info("Logged out.")
else:
    st.info("Log in to view protected content.")

with st.sidebar.expander("Security notes"):
    st.markdown(
        """
- Users are stored in `users.json` (demo). Use a database for real apps.
- Passwords are hashed with bcrypt in `auth_core.py`.
- `auth_core.py` is separated for unit testing.
        """
    )
