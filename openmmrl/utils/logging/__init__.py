import logging
import sys

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger instance.

    Args:
        name (str): Name of the logger, typically __name__.
        level (int): Logging level.

    Returns:
        logging.Logger: A configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if the logger is already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger 