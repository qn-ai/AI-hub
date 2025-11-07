# pages/login.py
import os
import time
import logging
from pathlib import Path

import streamlit as st

from util.log_config import setup_logging
from auth.auth_core import (
    load_users,
    save_users,
    register_user,
    can_login,
    reset_password,
)

# ------------------- Config -------------------
st.set_page_config(page_title="Login", page_icon="🔐")
TAB_NAMES = ["Login", "Register", "Reset Password"]
SESSION_TIMEOUT = 15 * 60  # 15 minutes
USERS_PATH = Path(os.getenv("USERS_JSON_PATH", "users.json"))

# ------------------- Logging -------------------
if "logging_initialized" not in st.session_state:
    logger = setup_logging(__name__)
    st.session_state["logging_initialized"] = True
else:
    logger = logging.getLogger(__name__)
logger.info("Login page loaded.")

# ------------------- Helpers -------------------
def setdefaults():
    for k, v in {
        "auth": False,
        "username": None,
        "name": None,
        "login_attempts": 0,
        "login_time": None,
    }.items():
        st.session_state.setdefault(k, v)

def flash(*msgs):
    """Queue sidebar messages to show on next render."""
    st.session_state["flash_sidebar_msgs"] = list(msgs)

def clear_auth():
    st.session_state.update({
        "auth": False,
        "username": None,
        "name": None,
        "login_attempts": 0,
        "login_time": None,
    })

def clear_inputs(*labels):
    """Remove specific input values stored by Streamlit."""
    for lbl in labels:
        st.session_state.pop(lbl, None)

def build_tabs(requested: str):
    """Return a dict of tab containers keyed by tab name.
    Streamlit opens the *first* tab by default, so we reorder."""
    order = [requested] + [t for t in TAB_NAMES if t != requested] \
            if requested in TAB_NAMES else TAB_NAMES
    t_containers = st.tabs(order)
    return dict(zip(order, t_containers))

# ------------------- Sidebar flash (render & clear) -------------------
with st.sidebar:
    msgs = st.session_state.pop("flash_sidebar_msgs", None)
    if msgs:
        for m in msgs:
            st.info(m)

# ------------------- Init / Data -------------------
setdefaults()
users = load_users(USERS_PATH)

# ------------------- Session timeout -------------------
if st.session_state["auth"] and st.session_state["login_time"]:
    if time.time() - st.session_state["login_time"] > SESSION_TIMEOUT:
        logger.info(f"Session timed out for user {st.session_state.get('username', '?')}")
        clear_auth()
        flash("Session timed out. Please log in again.")
        # We are already on login page; just rerun so Login tab opens
        st.session_state["wanted_tab"] = "Login"
        st.experimental_rerun()

# ------------------- Tabs -------------------
requested = st.session_state.pop("wanted_tab", "Login")
tabs = build_tabs(requested)

# =================== Login tab ===================
with tabs["Login"]:
    st.subheader("Login")

    if st.session_state["auth"]:
        st.caption(f"Logged in as **{st.session_state['name']}** "
                   f"({st.session_state['username']})")
        if st.button("Logout"):
            logger.info(f"User {st.session_state.get('username','?')} logged out")
            clear_auth()
            flash("You have been logged out.", "Please log in to use Holly app.")
            st.switch_page("home.py")
            st.stop()
    else:
        with st.form("login_form"):
            username = st.text_input(
                "Username",
                value=st.session_state.pop("prefill_username", "")
            )
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            st.session_state["login_attempts"] += 1
            logger.info(f"Login attempt #{st.session_state['login_attempts']} for user '{username}'")

            if st.session_state["login_attempts"] > 10:
                st.error("Too many attempts. Please reload.")
                logger.warning(f"Too many login attempts for user '{username}'")
            elif can_login(users, username, password):
                st.session_state.update({
                    "auth": True,
                    "username": username,
                    "name": users[username]["name"],
                    "login_attempts": 0,
                    "login_time": time.time(),
                })
                logger.info(f"User '{username}' logged in successfully.")
                flash(f"Welcome back, {st.session_state['name']}!", "You have been logged in successfully.")
                st.switch_page("home.py")
                st.stop()
            else:
                st.error("Invalid username or password.")
                logger.warning(f"Failed login attempt for user '{username}'")

# =================== Register tab ===================
with tabs["Register"]:
    st.subheader("Register")
    with st.form("register_form"):
        full_name = st.text_input("Full name")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account")

    if submitted:
        if not full_name or not new_username or not new_password:
            st.warning("All fields are required.")
        elif len(new_username) < 3:
            st.warning("Username must be at least 3 characters.")
        elif len(new_password) < 6:
            st.warning("Password must be at least 6 characters.")
        elif new_password != confirm_password:
            st.warning("Passwords do not match.")
            logger.warning(f"Password mismatch during registration for user '{new_username}'")
        elif new_username in users:
            st.warning("Username already exists.")
        else:
            try:
                register_user(users, new_username, full_name, new_password)
                save_users(USERS_PATH, users)
                logger.info(f"New user registered: '{new_username}' ({full_name})")

                # Clear inputs before switching tab
                clear_inputs("Full name", "Username", "Password", "Confirm password")

                # Flash + prefill + open Login tab on the same page
                st.session_state["prefill_username"] = new_username
                flash(f"User '{new_username}' created.", "Please log in now.")
                st.session_state["wanted_tab"] = "Login"
                st.experimental_rerun()
            except Exception as e:
                st.error("Registration failed.")
                logger.error(f"Registration error for user '{new_username}': {e}")

# =================== Reset Password tab ===================
with tabs["Reset Password"]:
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
            logger.warning(f"Password reset mismatch for user '{rp_username}'")
        else:
            try:
                reset_password(users, rp_username, old_password, new_password)
                save_users(USERS_PATH, users)
                logger.info(f"Password reset successful for user '{rp_username}'")

                # Clear inputs, flash, open Login tab (same page)
                clear_inputs("Username", "Current password", "New password", "Confirm new password")
                st.session_state["prefill_username"] = rp_username
                flash(f"Password reset successful for user '{rp_username}'.", "Please log in now.")
                st.session_state["wanted_tab"] = "Login"
                st.experimental_rerun()
            except Exception as e:
                st.error("Password reset failed.")
                logger.error(f"Password reset error for user '{rp_username}': {e}")
