"""
logging_config.py

Shared logging configuration for the Smart City Traffic Intelligence
capstone (Part 2). This is called exactly once, only from the __main__
block of whichever script is run directly — never from library code —
so that a single console + file handler pair on the root logger picks
up log messages from every module (pipeline, feature_engineering, etc.)
without each module configuring its own handlers.
"""

import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent / "pipeline.log"


def configure_logging(log_file=LOG_FILE):
    """Attach a console handler (INFO+) and a file handler (DEBUG+) to the
    root logger. Safe to call more than once — only configures handlers
    the first time, so re-imports or repeated calls never duplicate them."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    if root_logger.handlers:
        return

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Third-party libraries also use logging.getLogger(__name__), and their
    # loggers propagate up to the root logger by default. Since root is set
    # to DEBUG, that would otherwise flood pipeline.log with library-internal
    # noise (matplotlib's font_manager is especially verbose at DEBUG level).
    # Raising these to WARNING keeps the log file focused on this project's
    # own pipeline/feature_engineering/visualizations/CLI events.
    for noisy_logger in ("matplotlib", "PIL"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
