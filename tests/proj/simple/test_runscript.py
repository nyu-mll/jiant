import os
import pytest

import jiant.utils.python.io as py_io
from jiant.proj.simple import runscript as run
import jiant.scripts.download_data.runscript as downloader


@pytest.mark.parametrize("task_name", ["copa"])
@pytest.mark.parametrize("model_type", ["bert-base-cased"])
def test_simple_runscript(tmpdir, task_name, model_type):
    RUN_NAME = f"{test_simple_runscript.__name__}_{task_name}_{model_type}"
    data_dir = str(tmpdir.mkdir("data"))
    exp_dir = str(tmpdir.mkdir("exp"))

    downloader.download_data([task_name], data_dir)

    args = run.RunConfiguration(
        run_name=RUN_NAME,
        exp_dir=exp_dir,
        data_dir=data_dir,
        model_type=model_type,
        tasks=task_name,
        train_examples_cap=16,
        train_batch_size=16,
        no_cuda=True,
    )
    run.run_simple(args)

    val_metrics = py_io.read_json(os.path.join(exp_dir, "runs", RUN_NAME, "val_metrics.json"))
    assert val_metrics["aggregated"] > 0
