import itertools
import multiprocessing
import os
import time
from multiprocessing import Array, Value
from typing import Any, Dict, List, Tuple, Union
import shutil
import traceback
import sys
from contextlib import contextmanager
import signal
import numpy as np
from evalplus.eval.utils import (
    # create_tempdir,
    # reliability_guard,
    swallow_io,
    time_limit,
)
from multiprocessing import Array, Value
import resource

def set_memory_limit(limit_type, maximum_memory_bytes):
    """Sets the memory limits for the current process."""
    soft, hard = resource.getrlimit(limit_type)
    # Ensure we are not increasing the limit
    if maximum_memory_bytes < soft:
        soft = maximum_memory_bytes
    resource.setrlimit(limit_type, (soft, hard))

def get_memory_limits():
    """Gets the current memory limits for the process."""
    as_limit = resource.getrlimit(resource.RLIMIT_AS)
    data_limit = resource.getrlimit(resource.RLIMIT_DATA)
    return as_limit, data_limit

def reliability_guard(maximum_memory_bytes=None):
    """
    Limits memory usage for a single function call and then restores original memory limits.
    
    WARNING
    This function is NOT a security sandbox. Untrusted code should not be executed
    without a proper security sandbox.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get current limits
            original_as_limit, original_data_limit = get_memory_limits()
            
            try:
                if maximum_memory_bytes is not None:
                    # Set new memory limits without exceeding the original limits
                    set_memory_limit(resource.RLIMIT_AS, maximum_memory_bytes)
                    set_memory_limit(resource.RLIMIT_DATA, maximum_memory_bytes)
                
                # Execute the function
                return func(*args, **kwargs)
            
            finally:
                # Restore original memory limits
                resource.setrlimit(resource.RLIMIT_AS, original_as_limit)
                resource.setrlimit(resource.RLIMIT_DATA, original_data_limit)

        return wrapper
    
    return decorator



def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


class TimeoutException(Exception):
    pass

class TimeLimit:
    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        # Set the signal handler and the alarm
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_value, traceback):
        # Disable the alarm
        signal.alarm(0)
        # If an exception was raised, we need to propagate it
        if exc_type is TimeoutException:
            return False  # Propagate the exception
        return True  # Suppress other exceptions

    def handle_timeout(self, signum, frame):
        raise TimeoutException("Time limit exceeded")

# Context manager to redirect stdout and stderr
@contextmanager
def swallow_io():
    class DevNull:
        def write(self, _): pass
        def flush(self): pass

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = DevNull(), DevNull()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

# Example implementation of a context manager to temporarily change and restore directory
@contextmanager
def temp_directory():
    original_dir = os.getcwd()
    try:
        temp_dir = os.path.join(original_dir, "tempdir")
        os.makedirs(temp_dir, exist_ok=True)
        os.chdir(temp_dir)
        yield temp_dir
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

def execute_code(entry_point, code, inputs, expected, time_limits, atol, stat, details, progress, debug=False):
    unsafe_execute(entry_point, code, inputs, expected, time_limits, atol, stat, details, progress, debug)

@reliability_guard(maximum_memory_bytes=1024*1024*1024*1)
def unsafe_execute(
    entry_point: str,
    code: str,
    inputs: List,
    expected: List,
    time_limits: List,
    atol: float,
    stat,  # Value
    details,  # Array
    progress,  # Value
    debug=True
):
    if debug:
        print("Starting execution")

    exec_globals = {}
    original_stdout, original_stderr = sys.stdout, sys.stderr
    original_chdir = os.chdir
    og_putenv = os.putenv
    og_resource = sys.modules["resource"]

    maximum_memory_bytes = 1 * 1024 * 1024 * 1024
    # reliability_guard(maximum_memory_bytes=maximum_memory_bytes)
    # sys.stdout, sys.stderr = original_stdout, original_stderr
    # os.chdir = original_chdir
    # os.putenv = og_putenv
    # sys.modules["resource"] = og_resource
    # return stat, details
    try:
        with swallow_io():
            exec(code, exec_globals)
            
            fn = exec_globals.get(entry_point, None)
            if fn is None:
                if debug:
                    print(f"Function {entry_point} not found.")
                stat = 0
                return stat, details

        for i, inp in enumerate(inputs):
            try:
                with TimeLimit(time_limits[i]):
                    with swallow_io():
                        out = fn(*inp)
                        
                exp = expected[i]
                exact_match = out == exp
                print(out, exp, exact_match)
                if atol == 0 and is_float(exp):
                    atol = 1e-6  # enforce atol for float comparison
                if not exact_match and atol != 0:
                    assert type(out) == type(exp)
                    if isinstance(exp, (list, tuple)):
                        assert len(out) == len(exp)
                    assert np.allclose(out, exp, rtol=1e-07, atol=atol)
                else:
                    assert exact_match
            except Exception as e:
                print(")))", debug)
                if debug:
                    print(f"Exception during test {i}: {e}")
                # traceback.print_exc()
                details[i] = False
                progress += 1
                continue

            details[i] = True
            progress += 1

        stat = 1

    except Exception as e:
        if debug:
            print("Error during execution")
        traceback.print_exc()
        stat = 1

    finally:
        # Restore original state if necessary
        sys.stdout, sys.stderr = original_stdout, original_stderr
        os.chdir = original_chdir
        os.putenv = og_putenv
        sys.modules["resource"] = og_resource
        # Any additional cleanup can go here

    print(stat, details, "code")
    return stat, details



if __name__ == "__main__":
    code = '''def fib4(n: int) -> int:
    if n == 0: return 0
    elif n == 1: return 0
    elif n == 2: return 2
    elif n == 3: return 0

    a, b, c, d = 0, 0, 2, 0
    for _ in range(4, n + 1):
        a, b, c, d = b, c, d, a + b + c + d

    return d
    '''

    # _SUCCESS = 1
    # _FAILED = 0
    
    # inputs = [(0,), (1,), (5,)]
    # progress = 0
    # stat = 0
    # details = [False for _ in range(len(inputs))]

    
    # _, y = unsafe_execute(entry_point="fib4", code=code, inputs=inputs, expected=[0, 0, 4], time_limits=[60, 60, 60], atol=1e-6, stat=stat, details=details, progress=progress)
    # print(y)
    
    fn_name = "fib4"
    inputs = [(0,), (1,), (5,)]

    outputs = [0, 0, 4]
    progress = 0
    stat = 0
    details = [False for _ in range(len(inputs))]
    details = Array("b", [False for _ in range(len(inputs))])
    time_limits = [5 for _ in range(len(inputs))]

    # try:
    #     code = response.split("```python\n")[1].split("```")[0].split("assert")[0].split("# Test")[0].split("# Unit")[0].strip()
    # except:
    #     code = response.replace("# Your codes here\n", "").split("```")[0].strip()

    p = multiprocessing.Process(
    target=execute_code,
    args=(
        fn_name,
        code,
        inputs,
        outputs,
        time_limits,
        1e-6,
        stat,
        details,
        progress,
    ),
    )
    p.start()
    timeout = sum(time_limits)
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    print(stat, "!!!!!!!!!")
    print(details)
    result = all(details)

    response = '''```python
# Define the function
def double_every_other(numbers):
    # Initialize a variable to keep track of the current position
    current_position = 0

    # Create a copy of the input list to avoid modifying the original
    result = numbers[:]

    # Iterate through the list, starting from the left
    while current_position < len(result):
        # Double the value at the current position if it's even
        if result[current_position] % 2 == 0:
            result[current_position] *= 2

        # Move to the next position
        current_position += 2

    # Return the modified list
    return result
```<|eot_id|>'''

    try:
        code = response.split("```python\n")[1].split("```")[0].split("assert")[0].split("# Test")[0].split("# Unit")[0].strip()
    except:
        code = response.replace("# Your codes here\n", "").split("```")[0].strip()
    
    print("-"*50)
    print(code)
    print("-"*50)

    fn_name = "double_every_other"
    inputs = [[[0,1,2]]]

    outputs = [[0,1,4]]
    progress = 0
    stat = 0
    details = [False for _ in range(len(inputs))]
    details = Array("b", [False for _ in range(len(inputs))])
    time_limits = [5 for _ in range(len(inputs))]

    # try:
    #     code = response.split("```python\n")[1].split("```")[0].split("assert")[0].split("# Test")[0].split("# Unit")[0].strip()
    # except:
    #     code = response.replace("# Your codes here\n", "").split("```")[0].strip()

    p = multiprocessing.Process(
    target=execute_code,
    args=(
        fn_name,
        code,
        inputs,
        outputs,
        time_limits,
        1e-6,
        stat,
        details,
        progress,
        True
    ),
    )
    p.start()
    timeout = sum(time_limits)
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    print(stat, "!!!!!!!!!")
    print(details, [x for x in details])
    result = all(details)

    response = '''def sort_array(arr):
    odds = [num for num in arr if num % 2 != 0]
    evens = [num for num in arr if num % 2 == 0]
    sorted_odds = sorted(odds)
    result = []
    for num in arr:
        if num % 2 != 0:
            result.append(sorted_odds.pop(0))
        else:
            result.append(num)
    return result'''


    try:
        code = response.split("```python\n")[1].split("```")[0].split("assert")[0].split("# Test")[0].split("# Unit")[0].strip()
    except:
        code = response.replace("# Your codes here\n", "").split("```")[0].strip()
    
    print("-"*50)
    print(code)
    print("-"*50)

    fn_name = "sort_array"
    inputs = [[[5, 3, 2, 4, 1]]]

    outputs = [[1, 3, 2, 4, 5]]
    progress = 0
    stat = 0
    details = [False for _ in range(len(inputs))]
    details = Array("b", [False for _ in range(len(inputs))])
    time_limits = [5 for _ in range(len(inputs))]

    # try:
    #     code = response.split("```python\n")[1].split("```")[0].split("assert")[0].split("# Test")[0].split("# Unit")[0].strip()
    # except:
    #     code = response.replace("# Your codes here\n", "").split("```")[0].strip()

    p = multiprocessing.Process(
    target=execute_code,
    args=(
        fn_name,
        code,
        inputs,
        outputs,
        time_limits,
        1e-6,
        stat,
        details,
        progress,
        True
    ),
    )
    p.start()
    timeout = sum(time_limits)
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    print(stat, "!!!!!!!!!")
    print(details, [x for x in details])
    result = all(details)
    print(result)