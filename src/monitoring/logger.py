import logging
import sys
import time
from functools import wraps

from src.config import LOG_LEVEL, ENVIRONMENT

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format=f"%(asctime)s [{ENVIRONMENT}] %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def timed(logger: logging.Logger):
    """Decorator: logs duration and success/failure of any function. Applied
    to agent calls so you get real timing/error visibility instead of a
    silent black box.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                logger.info(f"{fn.__name__} succeeded in {duration_ms:.1f}ms")
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(f"{fn.__name__} failed after {duration_ms:.1f}ms: {e}")
                raise
        return wrapper
    return decorator
