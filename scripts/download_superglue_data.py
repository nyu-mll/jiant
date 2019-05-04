""" Script for downloading all SuperGLUE data.

For licence information, see the original dataset information links
available from: https://super.gluebenchmark.com/

Example usage:
  python download_superglue_data.py --data_dir data --tasks all

"""

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

TASKS = ["CB", "COPA", "MultiRC", "RTE", "WiC", "WSC", "diagnostic"]
TASK2PATH = {
    "CB": "https://dl.fbaipublicfiles.com/glue/superglue/data/CB.zip",
    "COPA": "https://dl.fbaipublicfiles.com/glue/superglue/data/COPA.zip",
    "MultiRC": "https://dl.fbaipublicfiles.com/glue/superglue/data/MultiRC.zip",
    "RTE": "https://dl.fbaipublicfiles.com/glue/superglue/data/RTE.zip",
    "WiC": "https://dl.fbaipublicfiles.com/glue/superglue/data/WiC.zip",
    "WSC": "https://dl.fbaipublicfiles.com/glue/superglue/data/WSC.zip",
    "diagnostic": "https://www.dropbox.com/s/ju7d95ifb072q9f/diagnostic-full.tsv?dl=1",
}


def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    if not os.path.isdir(os.path.join(data_dir, task)):
        os.mkdir(os.path.join(data_dir, task))
    data_file = os.path.join(data_dir, task, "%s.zip" % task)

    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(os.path.join(data_dir, task))
    os.remove(data_file)
    print("\tCompleted!")


def download_diagnostic(data_dir):
    print("Downloading and extracting diagnostic...")
    if not os.path.isdir(os.path.join(data_dir, "RTE")):
        os.mkdir(os.path.join(data_dir, "RTE"))
    data_file = os.path.join(data_dir, "RTE", "diagnostic-full.tsv")
    urllib.request.urlretrieve(TASK2PATH["diagnostic"], data_file)
    print("\tCompleted!")
    return


def get_tasks(task_names):
    task_names = task_names.split(",")
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
        if "RTE" in tasks and "diagnostic" not in tasks:
            tasks.append("diagnostic")
    return tasks


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", help="directory to save data to", type=str, default="../superglue_data"
    )
    parser.add_argument(
        "-t",
        "--tasks",
        help="tasks to download data for as a comma separated string",
        type=str,
        default="all",
    )
    args = parser.parse_args(arguments)

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        if task == "diagnostic":
            download_diagnostic(args.data_dir)
        else:
            download_and_extract(task, args.data_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
