# Task-specific setup

## UDPOS (XTREME)

UDPOS requires a specific version `networkx` to download. You can install it via

```bash
pip install networkx==1.11
```


## PAN-X (XTREME)

To preprocess PAN-X, you actually first need to download the file from: https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN.

The file should be named `AmazonPhotos.zip`, and it should be placed in `${task_data_base_path}/panx_temp/AmazonPhotos.zip` before running the download script.


## Bucc2018, Tatoeba (XTREME)

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
