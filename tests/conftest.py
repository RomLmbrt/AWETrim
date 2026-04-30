import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def pytest_sessionfinish(session, exitstatus):
    """Exit with 0 if no tests were collected.
    
    This hook is called after the entire test run is finished, and is used to
    override the exit status of the test run. If no tests were collected, the
    exit status is set to 0 (indicating success) instead of 5 (indicating no
    tests were collected). Otherwise, the template repository test actionwould 
    be marked as failing even though no tests were run.
    """
    if exitstatus == 5:  # no tests collected
        session.exitstatus = 0
