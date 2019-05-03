# jiant

[![CircleCI](https://circleci.com/gh/nyu-mll/jiant/tree/master.svg?style=svg)](https://circleci.com/gh/nyu-mll/jiant/tree/master)

`jiant` is a work-in-progress software toolkit for natural language processing research, designed to facilitate work on multitask learning and transfer learning for sentence understanding tasks.

A few things you might want to know about `jiant`:

- `jiant` is configuration-driven. You can run an enormous variety of experiments by simply writing configuration files. Of course, if you need to add any major new features, you can also easily edit or extend the code.
- `jiant` contains implementations of strong baselines for the [GLUE](https://gluebenchmark.com) and [SuperGLUE](https://super.gluebenchmark.com/) benchmarks, and it's the recommended starting point for work on these benchmarks.
- `jiant` was developed at [the 2018 JSALT Workshop](https://www.clsp.jhu.edu/workshops/18-workshop/) by [the General-Purpose Sentence Representation Learning](https://jsalt18-sentence-repl.github.io/) team and is maintained by [the NYU Machine Learning for Language Lab](https://wp.nyu.edu/ml2/people/), with help from [many outside collaborators](https://github.com/nyu-mll/jiant/graphs/contributors) (especially Google AI Language's [Ian Tenney](https://ai.google/research/people/IanTenney)).
- `jiant` is built on [PyTorch](https://pytorch.org). It also uses many components from [AllenNLP](https://github.com/allenai/allennlp) and HuggingFace PyTorch [implementations](https://github.com/huggingface/pytorch-pretrained-BERT) of BERT and GPT.
- The name `jiant` doesn't mean much. The 'j' stands for JSALT. That's all the acronym we have.


## Getting Started

1. Clone `jiant` and its submodules.
2. Check out our tutorial [here]()! 

### Submodules

This project uses [git submodules](https://blog.github.com/2016-02-01-working-with-submodules/) to manage some dependencies on other research code, in particular for loading CoVe and the OpenAI transformer model. In order to make sure you get these repos when you download `jiant/`, add `--recursive` to your clone command:

```sh
git clone --recursive git@github.com:jsalt18-sentence-repl/jiant.git jiant
```

If you already cloned and just need to get the submodules, you can run:
```sh
git submodule update --init --recursive
```

### Dependencies

Make sure you have installed the packages listed in `environment.yml`.
When listed, specific particular package versions are required.
If you use `conda` (recommended, [instructions for installing `miniconda` here](https://conda.io/miniconda.html)), you can create an environment from this package with the following command:

```
conda env create -f environment.yml
```

To activate the environment run ``source activate jiant``, and to deactivate run ``source deactivate``

Some requirements may only be needed for specific configurations. If you have trouble installing a specific dependency and suspect that it isn't needed for your use case, create an issue or a pull request, and we'll help you get by without it.

You will also need to install dependencies for nltk if you do not already have them:
```
python -m nltk.downloader -d /usr/share/nltk_data perluniprops nonbreaking_prefixes punkt
```


### Downloading data

The repo contains a convenience python script for downloading all [GLUE](https://gluebenchmark.com/) data and standard splits.

```
python scripts/download_glue_data.py --data_dir data --tasks all
```

We also make use of many other data sources, including:

- Translation: WMT'14 EN-DE, WMT'17 EN-RU. Scripts to prepare the WMT data are in [`scripts/wmt/`](scripts/wmt/).
- Language modeling: [Billion Word Benchmark](http://www.statmt.org/lm-benchmark/), [WikiText103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset). We use the English sentence tokenizer from [NLTK toolkit](https://www.nltk.org/) [Punkt Tokenizer Models](http://www.nltk.org/nltk_data/) to preprocess WikiText103 corpus. Note that it's only used in breaking paragraphs into sentences. It will use default tokenizer on word level as all other tasks unless otherwise specified. We don't do any preprocessing on BWB corpus.  
- Image captioning: MSCOCO Dataset (http://cocodataset.org/#download). Specifically we use the following splits: 2017 Train images [118K/18GB], 2017 Val images [5K/1GB], 2017 Train/Val annotations [241MB].
- Reddit: [reddit_comments dataset](https://bigquery.cloud.google.com/dataset/fh-bigquery:reddit_comments). Specifically we use the 2008 and 2009 tables.
- DisSent: Details for preparing the corpora are in [`scripts/dissent/README`](scripts/dissent/README).
- DNC (**D**iverse **N**atural Language Inference **C**ollection), i.e. recast data: The DNC is available [online](https://github.com/decompositional-semantics-initiative/DNC). Follow the instructions described there to download the DNC.
- CCG: Details for preparing the corpora are in [`scripts/ccg/README`](scripts/ccg/README).
- Edge probing analysis tasks: see _Papers_ below or [`probing/data`](probing/data/README.md) for more information.

To incorporate the above data, placed the data in the data directory in its own directory (see task-directory relations in `src/preprocess.py` and `src/tasks.py`.

### Running

To run an experiment, make a config file similar to `config/demo.conf` with your model configuration. You can use the `--overrides` flag to override specific variables. For example:
```sh
python main.py --config_file config/demo.conf \
    --overrides "exp_name = my_exp, run_name = foobar, d_hid = 256"
```
will run the demo config, but output to `$JIANT_PROJECT_PREFIX/my_exp/foobar`.

To run the demo config, you will have to set environment variables. The best way to achieve that is to follow the instructions in [user_config_template.sh](user_config_template.sh)
*  $JIANT_PROJECT_PREFIX: the where the outputs will be saved.
*  $JIANT_DATA_DIR: location of the saved data. This is usually the location of the GLUE data in a simple default setup.
*  $WORD_EMBS_FILE: location of any word embeddings you want to use (not necessary when using ELMo, GPT, or BERT). You can download GloVe (840B) [here](http://nlp.stanford.edu/data/glove.840B.300d.zip) or fastText (2M) [here](https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip).

To have `user_config.sh` run automatically, follow instructions in [scripts/export_from_bash.sh](export_from_bash.sh). 

## Command-Line Options

All model configuration is handled through the config file system and the `--overrides` flag, but there are also a few command-line arguments that control the behavior of `main.py`. In particular:

`--tensorboard` (or `-t`): use this to run a [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) server while the trainer is running, serving on the port specified by `--tensorboard_port` (default is `6006`).

The trainer will write event data even if this flag is not used, and you can run Tensorboard separately as:
```
tensorboard --logdir <exp_dir>/<run_name>/tensorboard
```

`--notify <email_address>`: use this to enable notification emails via [SendGrid](https://sendgrid.com/). You'll need to make an account and set the `SENDGRID_API_KEY` environment variable to contain the (text of) the client secret key.

`--remote_log` (or `-r`): use this to enable remote logging via Google Stackdriver. You can set up credentials and set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable; see [Stackdriver Logging Client Libraries](https://cloud.google.com/logging/docs/reference/libraries#client-libraries-usage-python).

## Saving Preprocessed Data

Because preprocessing is expensive (e.g. building vocab and indexing for very large tasks like WMT or BWB), we often want to run multiple experiments using the same preprocessing. So, we group runs using the same preprocessing in a single experiment directory (set using the ``exp_dir`` flag) in which we store all shared preprocessing objects. Later runs will load the stored preprocessing. We write run-specific information (logs, saved models, etc.) to a run-specific directory (set using flag ``run_dir``), usually nested in the experiment directory. Experiment directories are written in ``project_dir``. Overall the directory structure looks like:

```
project_dir  # directory for all experiments using jiant
|-- exp1/  # directory for a set of runs training and evaluating on FooTask and BarTask
|   |-- preproc/  # shared indexed data of FooTask and BarTask
|   |-- vocab/  # shared vocabulary built from examples from FooTask and BarTask
|   |-- FooTask/  # shared FooTask class object
|   |-- BarTask/  # shared BarTask class object
|   |-- run1/  # run directory with some hyperparameter settings
|   |-- run2/  # run directory with some different hyperparameter settings
|   |
|   [...]
|
|-- exp2/  # directory for a runs with a different set of experiments, potentially using a different branch of the code
|   |-- preproc/
|   |-- vocab/
|   |-- FooTask/
|   |-- BazTask/
|   |-- run1/
|   |
|   [...]
|
[...]
```

You should also set ``data_dir`` and  ``word_embs_file`` options to point to the directories containing the data (e.g. the output of the ``scripts/download_glue_data`` script) and word embeddings (optional, not needed when using ELMo, see later sections) respectively.

To force rereading and reloading of the tasks, perhaps because you changed the format or preprocessing of a task, delete the objects in the directories named for the tasks (e.g., `QQP/`) or use the option ``reload_tasks = 1``.

To force rebuilding of the vocabulary, perhaps because you want to include vocabulary for more tasks, delete the objects in `vocab/` or use the option ``reload_vocab = 1``.

To force reindexing of a task's data, delete some or all of the objects in `preproc/` or use the option ``reload_index = 1`` and set ``reindex_tasks`` to the names of the tasks to be reindexed, e.g. ``reindex_tasks=\"sst,mnli\"``. You should do this whenever you rebuild the task objects or vocabularies.

## Models (and how to add a Model)

The core model is a shared BiLSTM with task-specific components. When a language modeling objective is included in the set of training tasks, we use a bidirectional language model for all tasks, which is constructed to avoid cheating on the language modeling tasks. We also provide bag of words and RNN sentence encoder.

The base model class is a MultiTaskModel. To add another model, first add the class of the model to modules/modules.py, and then add the model construction in ``make_sent_encoder()`` (called in ``build_model()``) in src/models.py.

## Tutorials

To add a new task, refer to [INSERT LINK HERE]


## Suggested Citation

If you use `jiant` in academic work, please cite it directly:

```
@misc{wang2019jiant,
    author = {Alex Wang and Ian F. Tenney and Yada Pruksachatkun and Katherin Yu and Jan Hula and Patrick Xia and Raghu Pappagari and Shuning Jin and R. Thomas McCoy and Roma Patel and Yinghui Huang and Jason Phang and Edouard Grave and Najoung Kim and Phu Mon Htut and Thibault F'{e}vry and Berlin Chen and Nikita Nangia and Haokun Liu and and Anhad Mohananey and Shikha Bordia and Ellie Pavlick and Samuel R. Bowman},
    title = {{jiant} 0.9: A software toolkit for research on general-purpose text understanding models},
    howpublished = {\url{http://jiant.info/}},
    year = {2019}
}
```

## Papers

`jiant` has been used in these three papers so far:

- [Looking for ELMo's Friends: Sentence-Level Pretraining Beyond Language Modeling](https://arxiv.org/abs/1812.10860)
- [What do you learn from context? Probing for sentence structure in contextualized word representations](https://openreview.net/forum?id=SJzSgnRcKX) ("Edge Probing")
- [Probing What Different NLP Tasks Teach Machines about Function Word Comprehension](https://arxiv.org/abs/1904.11544)

To exactly reproduce experiments from [the ELMo's Friends paper](https://arxiv.org/abs/1812.10860) use the [`jsalt-experiments`](https://github.com/jsalt18-sentence-repl/jiant/tree/jsalt-experiments) branch. That will contain a snapshot of the code as of early August, potentially with updated documentation.

For the [edge probing paper](https://openreview.net/forum?id=SJzSgnRcKX), see the [probing/](probing/) directory.


## License

This package is released under the [MIT License](LICENSE.md). The material in the allennlp_mods directory is based on [AllenNLP](https://github.com/allenai/allennlp), which was originally released under the Apache 2.0 license.

## Getting Help

Post an issue here on GitHub if you have any problems, and create a pull request if you make any improvements (substantial or cosmetic) to the code that you're willing to share.

## FAQs

***It seems like my preproc/{task}\_\_{split}.data has nothing in it!***

This probably means that you probably ran the script before downloading the data for that task. Thus, delete the file from preproc and then run main.py again to build the data splits from scratch.

## Contributing

We use the `black` coding style with a line limit of 100. After installing the requirements, simply running `pre-commit
install` should ensure you comply with this in all your future commits. If you're adding features or fixing a bug,
please also add the tests.
