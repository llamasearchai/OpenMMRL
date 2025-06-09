import time
import functools
from openmmrl.utils.logging import get_logger

logger = get_logger(__name__)

def profile(func):
    """
    A simple decorator to profile the execution time of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute.")
        return result
    return wrapper 