import argparse
import json
import logging as log
import os
import random
import sys
import time
import types
import re
from typing import Iterable, Sequence, Type, Union

import pyhocon

from jiant.utils import hocon_writer


log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


class Params(object):
    """Params handler object.

    This functions as a nested dict, but allows for seamless dot-style access, similar to
    tf.HParams but without the type validation. For example:

    p = Params(name="model", num_layers=4)
    p.name  # "model"
    p['data'] = Params(path="file.txt")
    p.data.path  # "file.txt"
    """

    @staticmethod
    def clone(source, strict=True):
        if isinstance(source, pyhocon.ConfigTree):
            return Params(**source.as_plain_ordered_dict())
        elif isinstance(source, Params):
            return Params(**source.as_dict())
        elif isinstance(source, dict):
            return Params(**source)
        elif strict:
            raise ValueError("Cannot clone from type: " + str(type(source)))
        else:
            return None

    def __getitem__(self, k):
        return getattr(self, k)

    def __contains__(self, k):
        return k in self._known_keys

    def __setitem__(self, k, v):
        assert isinstance(k, str)
        if isinstance(self.get(k, None), types.FunctionType):
            raise ValueError("Invalid parameter name (overrides reserved name '%s')." % k)

        converted_val = Params.clone(v, strict=False)
        if converted_val is not None:
            setattr(self, k, converted_val)
        else:  # plain old data
            setattr(self, k, v)
        self._known_keys.add(k)

    def __delitem__(self, k):
        if k not in self:
            raise ValueError("Parameter %s not found.", k)
        delattr(self, k)
        self._known_keys.remove(k)

    def __init__(self, **kw):
        """Create from a list of key-value pairs."""
        self._known_keys = set()
        for k, v in kw.items():
            self[k] = v

    def regex_contains(self, k):
        """Searches Params for parameters that match a regex."""
        r = re.compile(k)
        results = list(filter(r.match, self._known_keys))
        return len(results) > 0

    def get(self, k, default=None):
        return getattr(self, k, default)

    def keys(self):
        return sorted(self._known_keys)

    def as_dict(self):
        """Recursively convert to a plain dict."""

        def convert(v):
            return v.as_dict() if isinstance(v, Params) else v

        return {k: convert(self[k]) for k in self.keys()}

    def __repr__(self):
        return self.as_dict().__repr__()

    def __str__(self):
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)


def get_task_attr(args: Type[Params], task_name: str, attr_name: str, default=None):
    """ Get a task-specific param.

    Look in args.task_name.attr_name, then fall back to default (if provided),
    then fall back to args.attr_name.
    """
    if task_name in args and (attr_name in args[task_name]):
        return args[task_name][attr_name]
    if default is not None:
        return default
    return args[attr_name]


def params_from_file(config_files: Union[str, Iterable[str]], overrides: str = None) -> Params:
    """Generate config map from config_files and overrides:

    1) read config file(s) lines (into a str)
    2) append overrides (into str from #1)
    3) call pyhocon's parse_string on combined config-str
    4) return a Params object (a custom jiant config map)

    Parameters
    ----------
    config_files : Union[str, Iterable[str]]
        filepath(s) for config files.
    overrides : str
        parameters overriding parameters found in config files.

    Returns
    -------
    Params
        config map.

    """
    config_string = ""
    if isinstance(config_files, str):
        config_files = [config_files]
    for config_file in config_files:
        with open(config_file) as fd:
            log.info("Loading config from %s", config_file)
            config_string += fd.read()
            config_string += "\n"
    if overrides:
        log.info("Config overrides: %s", overrides)
        # Append overrides to file to allow for references and injection.
        config_string += "\n"
        config_string += overrides
    basedir = os.path.dirname(config_file)  # directory context for includes
    config = pyhocon.ConfigFactory.parse_string(config_string, basedir=basedir)
    return Params.clone(config)


def write_params(params, config_file):
    config = pyhocon.ConfigFactory.from_dict(params.as_dict())
    with open(config_file, "w") as fd:
        fd.write(hocon_writer.HOCONConverter.to_hocon(config, indent=2))
