# Quick Start Guide — Using the command line interface
In this tutorial we'll show you how to do a basic multitask training experiment using `jiant`'s command line interface. This tutorial requires your machine to have a GPU supporting CUDA.
If you haven't already installed `jiant`, clone it and install its dependencies:
```
git clone https://github.com/jiant-dev/jiant.git
pip install -r jiant/requirements.txt
```
For this multitask training example we'll use MRPC and RTE tasks from GLUE, so we'll need to prepare the task data first: 
1. We'll get the data using Hugging Face's `download_glue_data.py` script:
```
wget https://raw.githubusercontent.com/huggingface/transformers/master/utils/download_glue_data.py
python download_glue_data.py \
    --data_dir ./raw_data \
    --tasks "MRPC,RTE"
```
2. We'll extract the data using `jiant`'s `export_glue_data.py` script:
```
export PYTHONPATH=jiant/
python jiant/jiant/scripts/preproc/export_glue_data.py \
    --input_base_path=./raw_data \
    --output_base_path=./tasks/ \
    --task_name_ls "mrpc,rte"
```
Now that the data is ready, we can use `jiant`'s simple CLI to perform multitask training with a single command:
```
python jiant/jiant/proj/simple/runscript.py \
    run \
    --run_name test_run \
    --exp_dir ./experiments/multi_task_mrpc_rte \
    --data_dir $(pwd)/tasks/data \
    --model_type bert-base-uncased \
    --train_batch_size 16 \
    --tasks mrpc,rte
```
This simple experiment showed that you've installed `jiant` and can run a basic multitask experiment. For more advanced experimental workflows see the example notebooks [here](../notebooks).  