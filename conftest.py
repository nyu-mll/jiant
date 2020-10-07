# -*- coding: utf-8 -*-
"""
Skipping slow tests according to command line: https://docs.pytest.org/en/latest/example/simple.html
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--rungpu", action="store_true", default=False, help="run gpu tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu: mark test as gpu required to run")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--rungpu"):
        skip_gpu = pytest.mark.skip(reason="need --rungpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
