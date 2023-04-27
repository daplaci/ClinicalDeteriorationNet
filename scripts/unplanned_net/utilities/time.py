import time
import functools

def timeit(level=1):
    def inner_timeit(method):
        """Print the runtime of the decorated function"""
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()  # 1
            value = method(*args, **kwargs)
            end_time = time.perf_counter()  # 2
            run_time = end_time - start_time  # 3
            level_str = "#".join(['' for _ in range(level+1)])
            print("\n{} Finished {} in {:.4f} secs ({} minutes )\n".format(level_str, method ,run_time, run_time//60))
            return value
        return wrapper_timer
    return inner_timeit
