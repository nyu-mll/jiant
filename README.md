# GLUE Baselines
This repo contains the code for baselines for the [Generalized Language Understanding Evaluation](https://gluebenchmark.com/) (GLUE) benchmark.
See [our paper](https://www.nyu.edu/projects/bowman/glue.pdf) for more details about GLUE or the baselines.

## Dependencies

Make sure you have the packages listed in environment.yml.
When listed, specific particular package versions required.
If you use conda, you can create an environment from this package with the following command:

```
conda env create -f environment.yml
```

## Downloading GLUE

We provide a convenience python script for downloading all GLUE data and standard splits.

```
python download_glue_data.py
```

## Running

To run our baselines, use either ``src/main.py`` or ``src/run_stuff.sh``.

```
python main.py --data_dir glue_data --tasks all
```

```
./run_stuff.sh -n -r -T all
```

## CoVe and ELMo

We use the CoVe implementation provided [here](https://github.com/salesforce/cove).
To use CoVe, clone the repo and fill in ``PATH_TO_COVE`` in ``src/models.py``.

We use the ELMo implementation provided by [AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md).

## Reference

If you use this code or GLUE, please consider citing us.

```
 @unpublished{wang2018glue
     title={{GLUE}: A Multi-Task Benchmark and Analysis Platform for
             Natural Language Understanding}
     author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill,
             Felix and Levy, Omer and Bowman, Samuel R.}
     note={arXiv preprint 1804.07461}
     year={2018}
 }
```


Feel free to contact alexwang _at_ nyu.edu with any questions or comments.
