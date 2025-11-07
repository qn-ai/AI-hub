# --- show sidebar flash on the login page, too ---
with st.sidebar:
    msgs = st.session_state.pop("flash_home_msgs", None)
    if msgs:
        for m in msgs:
            st.info(m)

# --- pick which tab should be first (Streamlit opens the first tab by default) ---
requested = st.session_state.pop("wanted_tab", "Login")
tab_order = ["Login", "Register", "Reset Password"]
if requested in tab_order:
    # put requested tab first so it opens by default
    tab_order = [requested] + [t for t in tab_order if t != requested]

# create tabs in computed order and map by name
t1, t2, t3 = st.tabs(tab_order)
tabs = {tab_order[0]: t1, tab_order[1]: t2, tab_order[2]: t3}

# use names to place your existing tab contents
with tabs["Login"]:
    # ... your login form ...

with tabs["Register"]:
    # ... your register form ...
    # On successful registration:
    # st.session_state["flash_home_msgs"] = [
    #     f"User '{new_username}' created.",
    #     "Please log in now."
    # ]
    # st.session_state["wanted_tab"] = "Login"
    # st.switch_page("pages/login.py")
    # st.stop()

with tabs["Reset Password"]:
    # ... your reset form ...
    # On successful reset:
    # st.session_state["flash_home_msgs"] = [
    #     f"Password reset successful for user '{username}'.",
    #     "Please log in now."
    # ]
    # st.session_state["wanted_tab"] = "Login"
    # st.switch_page("pages/login.py")
    # st.stop()
