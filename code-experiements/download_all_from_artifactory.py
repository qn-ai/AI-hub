with st.sidebar:
    # one-time sidebar messages
    msgs = st.session_state.pop("flash_sidebar_msgs", None)
    if msgs:
        for m in msgs:
            st.info(m)

    if st.session_state.get("auth"):
        st.caption(f"Logged in as **{st.session_state['name']}** ({st.session_state['username']})")
        if st.button("Logout", key="logout_sidebar"):
            clear = {"auth": False, "username": None, "name": None, "login_attempts": 0, "login_time": None}
            st.session_state.update(clear)
            st.session_state["flash_sidebar_msgs"] = ["You have been logged out.", "Please log in to use Holly app."]
            st.switch_page("home.py"); st.stop()
    else:
        if st.button("🔐 Login"):     st.session_state["wanted_tab"] = "Login"; st.switch_page("pages/login.py"); st.stop()
        if st.button("🧾 Register"):  st.session_state["wanted_tab"] = "Register"; st.switch_page("pages/login.py"); st.stop()
        if st.button("🔁 Reset Password"): st.session_state["wanted_tab"] = "Reset Password"; st.switch_page("pages/login.py"); st.stop()
