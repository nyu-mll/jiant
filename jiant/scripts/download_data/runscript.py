import os
import argparse

import jiant.utils.python.io as py_io
import jiant.scripts.download_data.datasets.nlp_tasks as nlp_tasks_download
import jiant.scripts.download_data.datasets.xtreme as xtreme_download
import jiant.scripts.download_data.datasets.files_tasks as files_tasks_download
from jiant.tasks.constants import GLUE_TASKS, SUPERGLUE_TASKS, XTREME_TASKS, BENCHMARKS

NLP_DOWNLOADER_TASKS = GLUE_TASKS | SUPERGLUE_TASKS
SUPPORTED_TASKS = NLP_DOWNLOADER_TASKS | XTREME_TASKS | {"squad_v1", "squad_v2"}


# noinspection PyUnusedLocal
def list_supported_tasks_cli(args):
    print("Supported tasks:")
    for task in SUPPORTED_TASKS:
        print(task)


def download_data_cli(args):
    output_base_path = args.output_path
    if args.tasks:
        task_names = args.tasks
    elif args.benchmark:
        task_names = BENCHMARKS[args.benchmark]
    else:
        raise RuntimeError()
    download_data(
        task_names=task_names, output_base_path=output_base_path,
    )


def download_data(task_names, output_base_path):
    task_data_base_path = py_io.create_dir(output_base_path, "data")
    task_config_base_path = py_io.create_dir(output_base_path, "configs")

    assert set(task_names).issubset(SUPPORTED_TASKS)

    # Download specified tasks and generate configs for specified tasks
    for i, task_name in enumerate(task_names):
        task_data_path = os.path.join(task_data_base_path, task_name)

        if task_name in NLP_DOWNLOADER_TASKS:
            nlp_tasks_download.download_data_and_write_config(
                task_name=task_name,
                task_data_path=task_data_path,
                task_config_path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
            )
        elif task_name in XTREME_TASKS:
            xtreme_download.download_xtreme_data_and_write_config(
                task_name=task_name,
                task_data_base_path=task_data_base_path,
                task_config_base_path=task_config_base_path,
            )
        elif task_name in ["squad_v1", "squad_v2"]:
            files_tasks_download.download_squad_data_and_write_config(
                task_name=task_name,
                task_data_path=task_data_path,
                task_config_path=os.path.join(task_config_base_path, f"{task_name}.json"),
            )
        else:
            raise KeyError()
        print(f"Downloaded and generated configs for '{task_name}' ({i+1}/{len(task_names)})")


def main():
    parser = argparse.ArgumentParser(description="Download NLP datasets and generate task configs")
    subparsers = parser.add_subparsers()
    sp_list = subparsers.add_parser("list", help="list supported tasks in downloader")
    sp_download = subparsers.add_parser("download", help="download data command")
    sp_download.add_argument(
        "--output_path", required=True, help="base output path for downloaded data and task configs"
    )
    sp_download_group = sp_download.add_mutually_exclusive_group(required=True)
    sp_download_group.add_argument("--tasks", nargs="+", help="list of tasks to download")
    sp_download_group.add_argument("--benchmark", choices=BENCHMARKS)

    # Hook subparsers up to functions
    sp_list.set_defaults(func=list_supported_tasks_cli)
    sp_download.set_defaults(func=download_data_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
