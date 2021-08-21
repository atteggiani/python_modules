def timer(func):
    """Print the runtime of the decorated function"""
    from functools import wraps
    import time
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer
    
def profiler(_func=None,outpath=None):
    """
    Print or save the profile output of the decorated function
    Usage: @profiler([,outpath=/path/to/output/file])
    """
    def profiler_inner(func):
        from functools import wraps
        import cProfile
        import pstats
        import sys
        @wraps(func)
        def wrapper_profiler(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            value = func(*args, **kwargs)
            profiler.disable()
            if outpath is None:
                stats = pstats.Stats(profiler)
                stats.strip_dirs()
                stats.print_stats()
            else:
                with open(outpath, 'w') as stream:
                    stats = pstats.Stats(profiler,stream=stream)
                    stats.strip_dirs()
                    stats.print_stats()      
            return value
        return wrapper_profiler
    return profiler_inner if _func is None else profiler_inner(_func)

def debug(func):
    """Print the function signature and return value"""
    from functools import wraps
    @wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug