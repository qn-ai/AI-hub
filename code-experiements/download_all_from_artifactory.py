def _reset_chat_state() -> None:
    """Clear chat-specific session state and flag confirmation."""
    logger = logging.getLogger(__name__)
    st.session_state["messages"] = []
    st.session_state["index"] = None
    st.session_state["embeddings"] = []
    st.session_state["input_tokens"] = None
    st.session_state["uploaded_files"] = []
    st.session_state["reset_chat_confirm"] = True
    logger.info("Chat state reset for user %s.", st.session_state.get("username"))
