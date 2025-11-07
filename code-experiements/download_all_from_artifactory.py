import streamlit as st
import logging
logger = logging.getLogger(__name__)

# ---- session defaults (safe to run every time) ----
for k, v in {"auth": False, "username": None, "name": None, "login_attempts": 0}.items():
    st.session_state.setdefault(k, v)

st.title("Home")

# ---------------- Sidebar (all-in-one) ----------------
with st.sidebar:
    st.write("Use the sidebar to navigate between different pages.")
    st.divider()

    # Flash messages (from login/logout/register/reset). One-time display.
    msgs = st.session_state.pop("flash_home_msgs", None)
    if msgs:
        for m in msgs:
            st.info(m)

    if st.session_state["auth"]:
        st.caption(
            f"Logged in as **{st.session_state['name']}** "
            f"({st.session_state['username']})"
        )
        if st.button("Logout", key="logout_sidebar"):
            logger.info(f"User {st.session_state.get('username','?')} clicked logout")

            # Clear auth/session
            st.session_state.update({
                "auth": False,
                "username": None,
                "name": None,
                "login_attempts": 0,
            })

            # Flash for next load (shown above in this sidebar)
            st.session_state["flash_home_msgs"] = [
                "You have been logged out.",
                "Please log in to use Holly app."
            ]

            # Redirect to Home (or wherever you want)
            st.switch_page("home.py")
            st.stop()
    else:
        if st.button("🔐 Login"):
            logger.info("Login button clicked.")
            st.switch_page("pages/login.py")
            st.stop()

# ---- main content (whatever you need below) ----
st.write("Welcome to Holly!")

### Login 

# ... inside your `if submitted` and successful auth branch ...
st.session_state.update({
    "auth": True,
    "username": username,
    "name": users[username]["name"],
    "login_attempts": 0,
})

# Flash for Home page (sidebar)
st.session_state["flash_home_msgs"] = [
    f"Welcome back, {st.session_state['name']}!",
    "You have been logged in successfully."
]

st.switch_page("home.py")
st.stop()


#### Logout

# clear
st.session_state.update({
    "auth": False,
    "username": None,
    "name": None,
    "login_attempts": 0,
})
# flash + redirect
st.session_state["flash_home_msgs"] = [
    "You have been logged out.",
    "Please log in to use Holly app."
]
st.switch_page("home.py")
st.stop()
