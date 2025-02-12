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

    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], object):
            self = args[0]
            func_name = f"{self.__class__.__name__}.{func.__name__}"
        else:
            self = None
            func_name = func.__name__

        TIMER_MANAGER = get_singleton_manager()
        func_name = f"{self.__class__.__name__}.{func.__name__}"

        if not TIMER_MANAGER.enable_logging:
            return func(*args, **kwargs)

        with TIMER_MANAGER.time(func_name):
            return func(*args, **kwargs)

    return wrapper
