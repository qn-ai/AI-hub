# logger.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str = "ml_pipeline") -> logging.Logger:
    """Create a module-specific logger with rotating file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = RotatingFileHandler(
            LOG_DIR / "pipeline.log",
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

    logger.propagate = False
    return logger
