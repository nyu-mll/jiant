## Task-specific Notes

### Adversarial NLI

[Adversarial NLI](https://arxiv.org/pdf/1910.14599.pdf) has 3 rounds of adversarial data creation. A1, A2 and A3 are different rounds of data creation. When downloading, you can use the task names `adversarial_nli_r1`, `adversarial_nli_r2`, `adversarial_nli_r3` to point the the different rounds. 

When doing training on the full ANLI dataset, which is SNLI+MNLI+A1+A2+A3, perform training in a multi-task manner with proportional sampling, and be sure to set the `task_to_taskmodel_map` to have all tasks point to the same NLI head.


### Masked Language Modeling (MLM)

MLM is a generic task, implemented with the `jiant_task_name` "`mlm_simple`". In other words, it is meant to be used with any appropriately formatted file.

`mlm_simple` expects input data files to be a single text file per phase, where each line corresponds to one example, and empty lines are ignored. This means that if a line corresponds to more than the `max_seq_length` of tokens during tokenization, everything past the first `max_seq_length` tokens per line will be ignored. We plan to add more complex implementations in the future.

You can structure your MLM task config file as follow:

```json
{
  "task": "mlm_simple",
  "paths": {
    "train": "/path/to/train.txt",
    "val": "/path/to/val.txt"
  },
  "name": "my_mlm_task"
}
```

### UDPOS (XTREME)

UDPOS requires a specific version `networkx` to download. You can install it via

```bash
pip install networkx==1.11
```


### PAN-X (XTREME)

To preprocess PAN-X, you actually first need to download the file from: https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN.

The file should be named `AmazonPhotos.zip`, and it should be placed in `${task_data_base_path}/panx_temp/AmazonPhotos.zip` before running the download script.


### Bucc2018, Tatoeba (XTREME)

The Bucc2018 and Tatoeba tasks are sentence retrieval tasks, and require the `faiss` library to run. `faiss-gpu` is recommended for speed reasons.

We recommend running:

```bash
conda install faiss-gpu cudatoolkit=10.1 -c pytorch
```

(Use the appropriate `cudatoolkit` version, which you can check with `nvcc --version`.)

Additionally, the task-model corresponding to retrieval tasks outputs an pooled embedding from a given layer of the encoder. As such, both the layer and pooling method need to be specified in taskmodel config. For instance, to replicate the baseline used in the XTREME benchmark, consider using:

```python
{
    "pooler_type": "mean",
    "layer": 14,
}
```

Also note that neither task has training sets, and Tatoeba does not have a separate test set.