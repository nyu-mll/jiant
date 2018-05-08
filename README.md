# GLUE Baselines
This repo contains the code for baselines for the [Generalized Language Understanding Evaluation](https://gluebenchmark.com/) (GLUE) benchmark.
See [our paper](https://www.nyu.edu/projects/bowman/glue.pdf) for more details about GLUE or the baselines.

## Dependencies

## Downloading GLUE

We provide a convenience python script for downloading all GLUE data and standard splits.

```
python download_glue_data.py
```

## Running

```
python main.py --data_dir glue_data --tasks all
```

```
./run_stuff.sh -n -r -T all
```

## CoVe and ELMo

We use the CoVe implementation provided [here](https://github.com/salesforce/cove).
To use CoVe, clone the repo and fill in ``PATH_TO_COVE`` in ```src/models.py``.

We use the ELMo implementation provided by [AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md).
