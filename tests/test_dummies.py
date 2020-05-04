# -*- coding: utf-8 -*-
"""Basic examples of tests. This includes:

1. test skipping
2. a simple, always-pass example
3. an example that is expected to fail
4. an example of a test that imports a module in this package

"""
import pytest
import torch
from jiant.demo import example_google


def test_gpu_only_test():
    if not torch.cuda.is_available():
        pytest.skip()


def test_passing_tests():
    assert True


@pytest.mark.xfail(raises=RuntimeError, reason="we wrote this test to demo an xfail")
def test_test_expected_to_fail():
    raise RuntimeError


def test_can_import_and_access_module_level_variable1():
    assert example_google.module_level_variable2 == 98765
