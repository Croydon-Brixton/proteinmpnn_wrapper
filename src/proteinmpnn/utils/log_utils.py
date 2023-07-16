import logging
import logging.handlers
import sys

from ..constants import DEFAULT_LOG_FORMATTER, DEFAULT_LOG_LEVEL


def _get_console_handler(
    stream=sys.stdout,
    formatter: logging.Formatter = DEFAULT_LOG_FORMATTER,
) -> logging.StreamHandler:
    """Returns Handler that prints to stdout."""
    console_handler = logging.StreamHandler(stream)
    console_handler.setFormatter(formatter)
    return console_handler


def get_logger(
    logger_name: str,
    level: int = DEFAULT_LOG_LEVEL,
    propagate: bool = False,
    log_to_console: bool = True,
    **handler_kwargs,
) -> logging.Logger:
    """Returns logger with console and timed file handler."""

    logger = logging.getLogger(logger_name)

    # if logger already has handlers attached to it, skip the configuration
    if logger.hasHandlers():
        logger.debug("Logger %s already set up.", logger.name)
        return logger

    logger.setLevel(level)

    if log_to_console:
        logger.addHandler(_get_console_handler(**handler_kwargs))

    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = propagate

    return logger
