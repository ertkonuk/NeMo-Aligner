import signal
from io import StringIO
from unittest.mock import patch
import sys

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

# set up signal handler for timeouts
signal.signal(signal.SIGALRM, timeout_handler)
timeout = 4 # seconds

def test_solution(solution_code:str, inputs: list, expected_output: list):
    pass