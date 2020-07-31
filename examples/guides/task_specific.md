# Task-specific setup

## Bucc2018, Tatoeba

The Bucc2018 and Tatoeba tasks are sentence retrieval tasks, and require the `faiss` library to run. `faiss-gpu` is recommended for speed reasons.

We recommend running:

```bash
conda install faiss-gpu cudatoolkit=10.1 -c pytorch
```

Additionally, the task-model corresponding to retrieval tasks outputs an pooled embedding from a given layer of the encoder. As such, both the layer and pooling method need to be specified in taskmodel config. For instance, to replicate the baseline used in the XTREME benchmark, consider using:

```python
{
    "pooler_type": "mean",
    "layer": 14,
}
```

Also note that neither task has training sets, and Tatoeba does not have a separate test set.
