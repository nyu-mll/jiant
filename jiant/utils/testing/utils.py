import sys


def is_pytest():
    return "pytest" in sys.modules
