import os

import jiant.utils.python.io as py_io


def write_configs(config_dict, base_path):
    os.makedirs(base_path, exist_ok=True)
    config_keys = [
        "task_config_path_dict",
        "task_cache_config_dict",
        "sampler_config",
        "global_train_config",
        "task_specific_configs_dict",
        "metric_aggregator_config",
    ]
    for path in config_dict["task_config_path_dict"].values():
        assert os.path.exists(path)
    for path_dict in config_dict["task_cache_config_dict"].values():
        for path in path_dict.values():
            assert os.path.exists(path)
    for config_key in config_keys:
        py_io.write_json(
            config_dict[config_key], os.path.join(base_path, f"{config_key}.json"),
        )
    py_io.write_json(config_dict, os.path.join(base_path, "full.json"))
    py_io.write_json(
        {
            f"{config_key}_path": os.path.join(base_path, f"{config_key}.json")
            for config_key in config_keys
        },
        path=os.path.join(base_path, "zz_full.json"),
    )
