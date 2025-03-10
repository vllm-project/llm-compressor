from functools import wraps

from tests.test_timer import Timer

__all__ = ["log_time", "get_singleton_manager"]


def get_singleton_manager(enable_logging: bool = True):
    """
    Return the Timer. If not has not yet been initialized, initialize and
    return. If it has, return the existing Timer.
    """
    if Timer._instance is None:
        Timer._instance = Timer(enable_logging=enable_logging)
    return Timer._instance


def log_time(func):
    """
    Decorator to time functions. Times for the function are stored using
    the class and function names.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        TIMER_MANAGER = get_singleton_manager()
        func_name = func.__name__

        if not TIMER_MANAGER.enable_logging:
            return func(*args, **kwargs)

        with TIMER_MANAGER.time(func_name):
            return func(*args, **kwargs)

    return wrapper
