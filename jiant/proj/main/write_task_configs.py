import os

import jiant.utils.python.filesystem as py_filesystem
import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf


def get_task_config(task_config_templates, task_name, task_data_dir):
    task_config = task_config_templates[task_name].copy()
    task_config["paths"] = {}
    for key, rel_path in task_config["rel_paths"].items():
        if isinstance(rel_path, dict):
            raise RuntimeError("Nested path dicts not currently supported")
        task_config["paths"][key] = os.path.join(task_data_dir, rel_path)
        assert os.path.exists(task_config["paths"][key])
    del task_config["rel_paths"]
    return task_config


def create_and_write_task_config(task_name, task_data_dir, task_config_path):
    task_config_templates = py_io.read_json(
        py_filesystem.get_code_asset_path("assets/simple_api/task_config_templates.json")
    )
    task_config = get_task_config(
        task_config_templates=task_config_templates,
        task_name=task_name,
        task_data_dir=task_data_dir,
    )
    os.makedirs(os.path.split(task_config_path)[0], exist_ok=True)
    py_io.write_json(task_config, task_config_path)


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    task_name = zconf.attr(type=str, required=True)
    task_data_dir = zconf.attr(type=str, required=True)
    task_config_path = zconf.attr(type=str, required=True)


def main(args: RunConfiguration):
    create_and_write_task_config(
        task_name=args.task_name,
        task_data_dir=args.task_data_dir,
        task_config_path=args.task_config_path,
    )


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
