import os
import shlex
import pytest

import jiant.utils.zconf as zconf


def get_json_path(file_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "jsons", file_name)


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    str_attr = zconf.attr()
    int_attr = zconf.attr(type=int)
    int_default_attr = zconf.attr(type=int, default=2)
    store_true_attr = zconf.attr(action="store_true")


def test_args_required_command_line():
    with pytest.raises(SystemExit):
        RunConfiguration.run_cli_json_prepend(cl_args=shlex.split(""))


def test_empty():
    args = RunConfiguration.run_cli_json_prepend(
        cl_args=shlex.split(
            f"""
        --ZZsrc={get_json_path("empty.json")}
        --str_attr "hi"
        --int_attr 1
    """
        )
    )
    assert args.str_attr == "hi"
    assert args.int_attr == 1
    assert args.int_default_attr == 2
    assert not args.store_true_attr


def test_simple():
    args = RunConfiguration.run_cli_json_prepend(
        cl_args=shlex.split(
            f"""
        --ZZsrc={get_json_path("simple.json")}
        --int_attr 1
    """
        )
    )
    assert args.str_attr == "hello"
    assert args.int_attr == 1
    assert args.int_default_attr == 3
    assert not args.store_true_attr


def test_simple_override_conflict():
    with pytest.raises(RuntimeError):
        RunConfiguration.run_cli_json_prepend(
            cl_args=shlex.split(
                f"""
            --ZZsrc={get_json_path("simple.json")}
            --str_attr "bye"
            --int_attr 1
        """
            )
        )


def test_simple_override_working():
    args = RunConfiguration.run_cli_json_prepend(
        cl_args=shlex.split(
            f"""
        --ZZsrc={get_json_path("simple.json")}
        --ZZoverrides str_attr
        --str_attr "bye"
        --int_attr 1
    """
        )
    )
    assert args.str_attr == "bye"
    assert args.int_attr == 1
    assert args.int_default_attr == 3
    assert not args.store_true_attr


def test_simple_double_override():
    args = RunConfiguration.run_cli_json_prepend(
        cl_args=shlex.split(
            f"""
        --ZZsrc={get_json_path("simple.json")}
        --ZZoverrides str_attr int_default_attr
        --str_attr "bye"
        --int_attr 1
        --int_default_attr 4
    """
        )
    )
    assert args.str_attr == "bye"
    assert args.int_attr == 1
    assert args.int_default_attr == 4
    assert not args.store_true_attr


def test_store_true():
    args = RunConfiguration.run_cli_json_prepend(
        cl_args=shlex.split(
            f"""
        --ZZsrc={get_json_path("store_true.json")}
        --str_attr "hello"
        --int_attr 1
    """
        )
    )
    assert args.str_attr == "hello"
    assert args.int_attr == 1
    assert args.int_default_attr == 2
    assert args.store_true_attr


def test_store_true_false():
    args = RunConfiguration.run_cli_json_prepend(
        cl_args=shlex.split(
            f"""
        --ZZsrc={get_json_path("store_true_false.json")}
        --str_attr "hello"
        --int_attr 1
    """
        )
    )
    assert args.str_attr == "hello"
    assert args.int_attr == 1
    assert args.int_default_attr == 2
    assert not args.store_true_attr


def test_store_true_override():
    args = RunConfiguration.run_cli_json_prepend(
        cl_args=shlex.split(
            f"""
        --ZZsrc={get_json_path("store_true.json")}
        --ZZoverrides store_true_attr
        --str_attr "hello"
        --int_attr 1
    """
        )
    )
    assert args.str_attr == "hello"
    assert args.int_attr == 1
    assert args.int_default_attr == 2
    assert not args.store_true_attr


def test_store_true_false_override():
    args = RunConfiguration.run_cli_json_prepend(
        cl_args=shlex.split(
            f"""
        --ZZsrc={get_json_path("store_true_false.json")}
        --ZZoverrides store_true_attr
        --str_attr "hello"
        --int_attr 1
        --store_true_attr
    """
        )
    )
    assert args.str_attr == "hello"
    assert args.int_attr == 1
    assert args.int_default_attr == 2
    assert args.store_true_attr
