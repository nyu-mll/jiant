import pytest

import jiant.utils.zconf as zconf


@zconf.run_config
class Config(zconf.RunConfig):
    attr1 = zconf.attr(default=None)
    attr2 = zconf.attr(required=True)


def test_args():
    config = Config(attr1=1, attr2=2)
    assert config.attr1 == 1
    assert config.attr2 == 2

    config = Config(attr2=2)
    assert config.attr1 is None
    assert config.attr2 == 2


def test_args_required():
    with pytest.raises(TypeError):
        Config()


def test_args_required_command_line():
    with pytest.raises(SystemExit):
        Config.run_cli_json_prepend(cl_args=[])


def test_to_dict():
    config = Config(attr1=1, attr2=2)
    conf_dict = config.to_dict()
    assert len(conf_dict) == 2
    assert conf_dict["attr1"] == 1
    assert conf_dict["attr2"] == 2
