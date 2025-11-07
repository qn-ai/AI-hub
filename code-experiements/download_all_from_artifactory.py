# --- sensible defaults (place once near top of page) ---
for k, v in {"auth": False, "username": None, "name": None, "login_attempts": 0}.items():
    st.session_state.setdefault(k, v)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.write("Use the sidebar to navigate between different pages.")
    st.divider()

    # Flash messages (one-time)
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
            logger.info(f"User {st.session_state.get('username','?')} logged out")

            # clear auth/session
            st.session_state.update({
                "auth": False,
                "username": None,
                "name": None,
                "login_attempts": 0,
            })

            # flash message for next page load (shown above on home)
            st.session_state["flash_home_msgs"] = [
                "You have been logged out.",
                "Please log in to use Holly app."
            ]

            st.switch_page("home.py")   # or "pages/login.py" if you prefer
            st.stop()                   # prevent further rendering after redirect
    else:
        if st.button("🔐 Login"):
            logger.info("Login button clicked.")
            st.switch_page("pages/login.py")
            st.stop()
