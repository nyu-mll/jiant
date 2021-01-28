import os
import pytest
import torch

import jiant.utils.python.io as py_io
from jiant.proj.simple import runscript as run
import jiant.scripts.download_data.runscript as downloader
import jiant.utils.torch_utils as torch_utils


@pytest.mark.gpu
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
        hf_pretrained_model_name_or_path=model_type,
        tasks=task_name,
        train_examples_cap=16,
        train_batch_size=16,
        no_cuda=True,
    )
    run.run_simple(args)

    val_metrics = py_io.read_json(os.path.join(exp_dir, "runs", RUN_NAME, "val_metrics.json"))
    assert val_metrics["aggregated"] > 0


@pytest.mark.gpu
@pytest.mark.parametrize("task_name", ["copa"])
@pytest.mark.parametrize("model_type", ["roberta-large"])
def test_simple_runscript_save(tmpdir, task_name, model_type):
    run_name = f"{test_simple_runscript.__name__}_{task_name}_{model_type}_save"
    data_dir = str(tmpdir.mkdir("data"))
    exp_dir = str(tmpdir.mkdir("exp"))

    downloader.download_data([task_name], data_dir)

    args = run.RunConfiguration(
        run_name=run_name,
        exp_dir=exp_dir,
        data_dir=data_dir,
        hf_pretrained_model_name_or_path=model_type,
        tasks=task_name,
        max_steps=1,
        train_batch_size=32,
        do_save=True,
        eval_every_steps=10,
        learning_rate=0.01,
        num_train_epochs=5,
    )
    run.run_simple(args)

    # check best_model and last_model exist
    assert os.path.exists(os.path.join(exp_dir, "runs", run_name, "best_model.p"))
    assert os.path.exists(os.path.join(exp_dir, "runs", run_name, "best_model.metadata.json"))
    assert os.path.exists(os.path.join(exp_dir, "runs", run_name, "last_model.p"))
    assert os.path.exists(os.path.join(exp_dir, "runs", run_name, "last_model.metadata.json"))

    # assert best_model not equal to last_model
    best_model_weights = torch.load(
        os.path.join(exp_dir, "runs", run_name, "best_model.p"), map_location=torch.device("cpu")
    )
    last_model_weights = torch.load(
        os.path.join(exp_dir, "runs", run_name, "last_model.p"), map_location=torch.device("cpu")
    )
    assert not torch_utils.eq_state_dicts(best_model_weights, last_model_weights)

    run_name = f"{test_simple_runscript.__name__}_{task_name}_{model_type}_save_best"
    args = run.RunConfiguration(
        run_name=run_name,
        exp_dir=exp_dir,
        data_dir=data_dir,
        hf_pretrained_model_name_or_path=model_type,
        tasks=task_name,
        max_steps=1,
        train_batch_size=16,
        do_save_best=True,
    )
    run.run_simple(args)

    # check only best_model saved
    assert os.path.exists(os.path.join(exp_dir, "runs", run_name, "best_model.p"))
    assert os.path.exists(os.path.join(exp_dir, "runs", run_name, "best_model.metadata.json"))
    assert not os.path.exists(os.path.join(exp_dir, "runs", run_name, "last_model.p"))
    assert not os.path.exists(os.path.join(exp_dir, "runs", run_name, "last_model.metadata.json"))

    # check output last model
    run_name = f"{test_simple_runscript.__name__}_{task_name}_{model_type}_save_last"
    args = run.RunConfiguration(
        run_name=run_name,
        exp_dir=exp_dir,
        data_dir=data_dir,
        hf_pretrained_model_name_or_path=model_type,
        tasks=task_name,
        max_steps=1,
        train_batch_size=16,
        do_save_last=True,
    )
    run.run_simple(args)

    # check only last_model saved
    assert not os.path.exists(os.path.join(exp_dir, "runs", run_name, "best_model.p"))
    assert not os.path.exists(os.path.join(exp_dir, "runs", run_name, "best_model.metadata.json"))
    assert os.path.exists(os.path.join(exp_dir, "runs", run_name, "last_model.p"))
    assert os.path.exists(os.path.join(exp_dir, "runs", run_name, "last_model.metadata.json"))
