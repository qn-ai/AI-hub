import logging
import streamlit as st

def setup_logging(name: str | None = None) -> logging.Logger:
    """Configure stdout logging once, inject username from session_state, no duplicates."""
    class SessionStateFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.username = st.session_state.get("username", "guest")
            return True

    logger = logging.getLogger(name) if name else logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False  # <- avoid double-logging via root handlers

    # Idempotent guard: if we already configured, return
    if getattr(logger, "_configured", False):
        return logger

    # Create ONE stream handler (or reuse existing one)
    existing_stream = next((h for h in logger.handlers
                            if isinstance(h, logging.StreamHandler)), None)
    if existing_stream is None:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(username)s - %(name)s - %(levelname)s - %(message)s'
        )
        stream_handler.setFormatter(formatter)
        # attach filter to BOTH logger (covers all handlers) and handler (defensive)
        f = SessionStateFilter()
        logger.addFilter(f)
        stream_handler.addFilter(f)
        logger.addHandler(stream_handler)
    else:
        # Ensure formatter/filter exist on the existing handler
        if not any(isinstance(f, SessionStateFilter) for f in logger.filters):
            logger.addFilter(SessionStateFilter())
        if not any(isinstance(f, SessionStateFilter) for f in existing_stream.filters):
            existing_stream.addFilter(SessionStateFilter())
        if existing_stream.formatter is None:
            existing_stream.setFormatter(logging.Formatter(
                '%(asctime)s - %(username)s - %(name)s - %(levelname)s - %(message)s'
            ))

    logger._configured = True
    logger.info("Logging initialized.")
    return logger
