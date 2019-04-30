# jiant
This repo contains the `jiant` sentence representation learning toolkit created at the [2018 JSALT Workshop](https://www.clsp.jhu.edu/workshops/18-workshop/) by the [General-Purpose Sentence Representation Learning](https://jsalt18-sentence-repl.github.io/) team. It is an extensible platform meant to make it easy to run experiments that involve multitask and transfer learning across sentence-level NLP tasks.

The 'j' in `jiant` stands for JSALT. That's all the acronym we have.

jiant has been used in these two papers so far:

- [Looking for ELMo's Friends: Sentence-Level Pretraining Beyond Language Modeling](https://arxiv.org/abs/1812.10860)
- [What do you learn from context? Probing for sentence structure in contextualized word representations](https://openreview.net/forum?id=SJzSgnRcKX) ("Edge Probing")

To exactly reproduce experiments from [the ELMo's Friends paper](https://arxiv.org/abs/1812.10860) use the [`jsalt-experiments`](https://github.com/jsalt18-sentence-repl/jiant/tree/jsalt-experiments) branch. That will contain a snapshot of the code as of early August, potentially with updated documentation.

For the [edge probing paper](https://openreview.net/forum?id=SJzSgnRcKX), see the [probing/](probing/) directory.

## Getting Started
1) Run the dependencies file.
2) Download the submodules.
3) Download the GLUE data.
4) Run demo.sh to see an example of pretraining and fine-tuning on a task.

## Dependencies

Make sure you have installed the packages listed in `environment.yml`.
When listed, specific particular package versions are required.
If you use conda (recommended, [instructions for installing miniconda here](https://conda.io/miniconda.html)), you can create an environment from this package with the following command:

```
conda env create -f environment.yml
```

To activate the environment run ``source activate jiant``, and to deactivate run ``source deactivate``

Some requirements may only be needed for specific configurations. If you have trouble installing a specific dependency and suspect that it isn't needed for your use case, create an issue or a pull request, and we'll help you get by without it.

You will also need to install dependencies for nltk if you do not already have them:
```
python -m nltk.downloader -d /usr/share/nltk_data perluniprops nonbreaking_prefixes punkt
```

## Submodules

This project uses [git submodules](https://blog.github.com/2016-02-01-working-with-submodules/) to manage some dependencies on other research code, in particular for loading CoVe and the OpenAI transformer model. In order to make sure you get these repos when you download `jiant/`, add `--recursive` to your clone command:

```sh
git clone --recursive git@github.com:jsalt18-sentence-repl/jiant.git jiant
```

If you already cloned and just need to get the submodules, you can do:
```sh
git submodule update --init --recursive
```

## Downloading data

The repo contains a convenience python script for downloading all [GLUE](https://www.nyu.edu/projects/bowman/glue.pdf) data and standard splits.

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
- Edge probing analysis tasks: see [`probing/data`](probing/data/README.md) for more information.

To incorporate the above data, placed the data in the data directory in its own directory (see task-directory relations in `src/preprocess.py` and `src/tasks.py`.

## Running

To run an experiment, make a config file similar to `config/demo.conf` with your model configuration. You can use the `--overrides` flag to override specific variables. For example:
```sh
python main.py --config_file config/demo.conf \
    --overrides "exp_name = my_exp, run_name = foobar, d_hid = 256"
```
will run the demo config, but output to `$JIANT_PROJECT_PREFIX/my_exp/foobar`.

To run the demo config, you will have to set environment variables. The best way to achieve that is to follow the instructions in [path_config.sh](path_config.sh)
*  $JIANT_PROJECT_PREFIX: the where the outputs will be saved.
*  $JIANT_DATA_DIR: location of the saved data. This is usually the location of the Glue data.
*  $WORD_EMBED: location of the word embeddings you want to use. For GloVe:  [840B300d Glove](http://nlp.stanford.edu/data/glove.840B.300d.zip). For FastText: [300d-2M](https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip). For ELMo, AllenNLP will download it for you. For OpenAI, the model weights will be downloaded when installing the git submodules.




### Saving Preprocessed Data

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

### Command-Line Options

All model configuration is handled through the config file system and the `--overrides` flag, but there are also a few command-line arguments that control the behavior of `main.py`. In particular:

`--tensorboard` (or `-t`): use this to run a [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) server while the trainer is running, serving on the port specified by `--tensorboard_port` (default is `6006`).

The trainer will write event data even if this flag is not used, and you can run Tensorboard separately as:
```
tensorboard --logdir <exp_dir>/<run_name>/tensorboard
```

`--notify <email_address>`: use this to enable notification emails via [SendGrid](https://sendgrid.com/). You'll need to make an account and set the `SENDGRID_API_KEY` environment variable to contain the (text of) the client secret key.

`--remote_log` (or `-r`): use this to enable remote logging via Google Stackdriver. You can set up credentials and set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable; see [Stackdriver Logging Client Libraries](https://cloud.google.com/logging/docs/reference/libraries#client-libraries-usage-python).

## Models (and how to add a Model)

The core model is a shared BiLSTM with task-specific components. When a language modeling objective is included in the set of training tasks, we use a bidirectional language model for all tasks, which is constructed to avoid cheating on the language modeling tasks. We also provide bag of words and RNN sentence encoder.

The base model class is a MultiTaskModel. To add another model, first add the class of the model to modules/modules.py, and then add the model construction in ``make_sent_encoder()`` (called in ``build_model()``) in src/models.py.

Task-specific components include logistic regression and multi-layer perceptron for classification and regression tasks, and an RNN decoder with attention for sequence transduction tasks.
To see the full set of available params, see [config/defaults.conf](config/defaults.conf). For a list of options affecting the execution pipeline (which configuration file to use, whether to enable remote logging or tensorboard, etc.), see the arguments section in [main.py](main.py).

To use the ON-LSTM sentence encoder from [Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks](https://arxiv.org/abs/1810.09536), set ``sent_enc = onlstm``. To re-run experiments from the paper on WSJ Language Modeling, use the configuration file [config/onlstm.conf](config/onlstm.conf). Specific ON-LSTM modules use code from the [Github](https://github.com/yikangshen/Ordered-Neurons) implementation of the paper.

To use the PRPN sentence encoder from [***Neural language modeling by jointly learning syntax and lexicon***](https://arxiv.org/abs/1711.02013), set ``sent_enc=prpn``. To re-run experiments from the paper on WSJ Language Modeling, use the configuration file [config/prpn.conf](config/prpn.conf). Specific PRPN modules use code from the [Github](https://github.com/yikangshen/PRPN) implementation of the paper.

## Currently Supported Task Types

We currently support the following:

	* Single sentence classification tasks
	* Pair sentence classification tasks
	* Regression tasks
	* Tagging tasks
	* Span classification Tasks - to run these, we currently require an extra preprocessing 
	  step, which consists of preprocessing the data to get BERT tokenized span indices. 
	  SpanTasks expects the files to be in json format and be named as {file_name}.retokenized.{tokenizer_name}.
	* seq2seq tasks are partially supported.

### Transformers 

We also include an experimental option to use a shared [Transformer](https://arxiv.org/abs/1706.03762) in place of the shared BiLSTM by setting ``sent_enc = transformer``. When using a Transformer, we use the [Noam learning rate scheduler](https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers.py#L84), as that seems important to training the Transformer thoroughly. 

We also support using pretrained Transformer language models. To use the OpenAI transformer model, set `openai_transformer = 1`, download the [model](https://github.com/openai/finetune-transformer-lm) folder that contains pre-trained models, and place it under `src/openai_transformer_lm/pytorch_huggingface/`.
To use [BERT](https://arxiv.org/abs/1810.04805) architecture, set ``bert_model_name`` to one of the models listed [here](https://github.com/huggingface/pytorch-pretrained-BERT#loading-google-ai-or-openai-pre-trained-weigths-or-pytorch-dump), e.g. ``bert-base-cased``. You should also set ``tokenizer`` to be the BERT model used in order to ensure you are using the same tokenization and vocabulary.

When using BERT, we follow the procedures set out in the original work as closely as possible: For pair sentence tasks, we concatenate the sentences with a sepcial `[SEP]` token. Rather than max-pooling, we take the first representation of the sequence (corresponding to the special `[CLS]` token) as the representation of the entire sequence.
We also have support for the version of Adam that was used in training BERT (``optimizer = bert_adam``).

## Trainer

The trainer was originally written to perform sampling-based multi-task training. At each step, a task is sampled and ``bpp_base`` (default: 1) batches of that task's training data is trained on.
The trainer evaluates the model on the validation data after a fixed number of gradient steps, set by ``val_interval``.
The learning rate is scheduled to decay by ``lr_decay_factor`` (default: .5) whenever the validation score doesn't improve after ``lr_patience`` (default: 1) validation checks.
Note: "epoch" is generally used in comments and variable names to refer to the interval between validation checks, not to a complete pass through any one training set.

If you're training only on one task, you don't need to worry about sampling schemes, but if you are training on multiple tasks, you can vary the sampling weights with ``weighting_method``, e.g. ``weighting_method = uniform`` or ``weighting_method = proportional`` (to amount of training data). You can also scale the losses of each minibatch via ``scaling_method`` if you want to weight tasks with different amounts of training data equally throughout training.

For multi-task training, we use a shared global optimizer and LR scheduler for all tasks. In the global case, we use the macro average of each task's validation metrics to do LR scheduling and early stopping. When doing multi-task training and at least one task's validation metric should decrease (e.g. perplexity), we invert tasks whose metric should decrease by averaging ``1 - (val_metric / dec_val_scale)``, so that the macro-average will be well-behaved.

We have partial support for per-task optimizers (``shared_optimizer = 0``), but checkpointing may not behave correctly in this configuration. In the per-task case, we stop training on a task when its patience has run out or its optimizer hits the minimum learning rate. 

Within a run, tasks are distinguished between training tasks (pretrain_tasks) and evaluation tasks (target tasks). The logic of ``main.py`` is that the entire model is pretrained on all the `pre_training` tasks, then the best model is then loaded, and task-specific components are trained for each of the evaluation tasks with a frozen shared sentence encoder.
You can control which steps are performed or skipped by setting the flags ``do_pretrain, do_target_task_training, do_full_eval``.
Specify training tasks with ``pretrain_tasks = $pretrain_tasks`` where ``$pretrain_tasks`` is a comma-separated list of task names; similarly use ``target_tasks`` to specify the eval-only tasks.
For example, ``pretrain_tasks = \"sst,mnli,foo\", target_tasks = \"qnli,bar,sst,mnli,foo\"`` (HOCON notation requires escaped quotes in command line arguments).
Note: if you want to train and evaluate on a task, that task must be in both ``pretrain_tasks`` and ``target_tasks``.

We support two modes of adapting pretrained models to target tasks. 
Setting `transfer_paradigm = finetune` will fine-tune the entire model while training for a target task.
The mode will create a copy of the model _per target task_.
Setting `transfer_paradigm = frozen` will only train the target-task specific components while training for a target task.
If using ELMo and `sep_embs_for_skip = 1`, we will also learn a task-specific set of layer-mixing weights.

## Adding New Tasks

To add new tasks, you should:

1. Add your data to the ``data_dir`` you intend to use. When constructing your task class (see next bullet), make sure you specify the correct subfolder containing your data.

2. Shuffle your training and validation data if there are many rows (>10k) to avoid training artefacts. Indeed, jiant loads 10k examples at a time in memory and then only it shuffles them. This could create issues if your data is sorted, e.g: If your data is sorted by label, the model would go through all the examples of a class before finding one of another, so it would learn to always predict the first class. If you reach 100% accuracy before one epoch, this is likely the case.

3. Create a class in ``src/tasks.py``, and make sure that...



    * You decorate : in the line immediately before ``class MyNewTask():``, add the line ``@register_task(task_name, rel_path='path/to/data')`` where ``task_name`` is the designation for the task used in ``pretrain_tasks, target_tasks`` and ``rel_path`` is the path to the data in ``data_dir``. See `EdgeProbingTasks` in [`tasks.py`](src/tasks/edge_probing.py) for an example.
    * Your task inherits from existing classes as necessary (e.g. ``PairClassificationTask``, ``SequenceGenerationTask``, ``WikiTextLMTask``, etc.).
    * The task definition includes the data loader, as a method called ``load_tokenized_data()`` which stores tokenized but un-indexed data for each split in attributes named ``task.{train,valid,test}_data_text``. The formatting of each datum can be anything as long as your preprocessing code (in ``src/preprocess.py``, see next bullet) expects that format. Generally data are formatted as lists of inputs and output, e.g. MNLI is formatted as ``[[sentences1]; [sentences2]; [labels]]`` where ``sentences{1,2}`` is a list of the first sentences from each example. Make sure to call your data loader in initialization!
    * Your task implements a method ``task.get_sentences()`` that iterates over all text to index in order to build the vocabulary. For some types of tasks, e.g. ``SingleClassificationTask``, you only need set ``task.sentences`` to be a list of sentences (``List[List[str]]``).
    * Your task implements a method ``task.count_examples()`` that sets ``task.example_counts`` (``Dict[str:int]``): the number of examples per split (train, val, test). See [here](https://github.com/jsalt18-sentence-repl/jiant/blob/master/src/tasks/tasks.py#L647) for an example.
    * Your task implements a method ``task.get_split_text()`` that takes in the name of a split and returns an iterable over the data in that split. This method will be called in preprocessing and passed to ``task.process_split`` (see next bullet).
    * Your task implements a method ``task.process_split()`` that takes in a split of your data and produces an iterable of AllenNLP ``Instance``s. An ``Instance`` is a wrapper around a dictionary of ``(field_name, Field)`` pairs. ``Field``s are objects to help with data processing (indexing, padding, etc.). Each input and output should be wrapped in a field of the appropriate type (``TextField`` for text, ``LabelField`` for class labels, etc.). For MNLI, we wrap the premise and hypothesis in ``TextField``s and the label in ``LabelField``. See the [AllenNLP tutorial](https://allennlp.org/tutorials) or the examples in ``src/tasks.py``.  The names of the fields, e.g. ``input1``, can be named anything so long as the corresponding code in ``src/models.py`` (see next bullet) expects that named field. However make sure that the values to be predicted are either named ``labels`` (for classification or regression) or ``targs`` (for sequence generation)!
    * If you task requires task specific label namespaces, e.g. for translation or tagging, you set the attribute ``task._label_namespace`` to reserve a vocabulary namespace for your task's target labels. We strongly suggest including the task name in the target namespace. Your task should also implement ``task.get_all_labels()``, which returns an iterable over the labels (possibly words, e.g. in the case of MT) in the task-specific namespace.
    * Your task has attributes ``task.val_metric`` (name of task-specific metric to track during training) and ``task.val_metric_decreases`` (bool, ``True`` if val metric should decrease during training). You should also implement a ``task.get_metrics()`` method that implements the metrics you care about by using AllenNLP ``Scorer`` objects (typically set via ``task.scorer1``, ``task.scorer2``, etc.).

3. In ``src/models.py``, make sure that:
    * The correct task-specific module is being created for your task in ``build_module()``, which adds the task-specific components of a model to the model being used. For example, for single classification task, a linear layer may be generated in build_module().
    * Your task is correctly being handled in ``forward()`` of ``MultiTaskModel``. The model will receive the task class you created and a batch of data, where each batch is a dictionary with keys of the ``Instance`` objects you created in preprocessing, as well as a ``predict`` flag that indicates if your forward function should generate predictions or not.
    * You create additional methods or add branches to existing methods as necessary. If you do add additional methods, make sure to make use of the ``sent_encoder`` attribute of the model, which is shared amongst all tasks.

Note: The current training procedure is task-agnostic: we randomly sample a task to train on, pass a batch to the model, and receive an output dictionary at least containing a ``loss`` key. Training loss should be calculated within the model; validation metrics should also be computed within AllenNLP ``scorer``s and not in the training loop. So you should *not* need to modify the training loop; please reach out if you think you need to.

Feel free to create a pull request to add an additional task if you expect that it'll be useful to others.

## Pretrained Embeddings

### ELMo

We use the ELMo implementation provided by [AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md).
To use ELMo, set ``elmo`` to 1.
By default, AllenNLP will download and cache the pretrained ELMo weights. If you want to use a particular file containing ELMo weights, set ``elmo_weight_file_path = path/to/file``.

To use only the _character-level CNN word encoder_ from ELMo by use `elmo_chars_only = 1`. _This is set by default_.


### CoVe

We use the CoVe implementation provided [here](https://github.com/salesforce/cove).
To use CoVe, clone the repo and set the option ``path_to_cove = "/path/to/cove/repo"`` and set ``cove = 1``.


### FastText

Download the pretrained vectors located [here](https://fasttext.cc/docs/en/english-vectors.html), preferrably the 300-dimensional Common Crawl vectors. Set the ``word_emb_file`` to point to the .vec file.


### GloVe

To use [GloVe pretrained word embeddings](https://nlp.stanford.edu/projects/glove/), download and extract the relevant files and set ``word_embs_file`` to the GloVe file.


## Quick-Start on GCP (for JSALT internal use only)

For the JSALT workshop, we used Google Compute Engine as our main compute platform. If you're using Google Compute Engine, the private project instance images (`cpu-workstation-template*` and `gpu-worker-template-*`) already have all the required packages installed, plus the GLUE data and pre-trained embeddings downloaded to `/usr/share/jsalt`. Unfortunately, these images are not straightforward to share. To use, clone this repo to your home directory, then test with:

```sh
python main.py --config_file config/demo.conf
```

You should see the model start training, and achieve an accuracy of > 70% on SST in a few minutes. The default config will write the experiment directory to `$HOME/exp/<experiment_name>` and the run directory to `$HOME/exp/<experiment_name>/<run_name>`, so you can find the demo output in `~/exp/jiant-demo/sst`.


## Update config files

As some config arguments are renamed, you may encounter an error when loading past config files (e.g. params.conf) created before Oct 24, 2018. To update a config file, run

```sh
python scripts/update_config.py <path_to_file>
```

## License

This package is released under the [MIT License](LICENSE.md). The material in the allennlp_mods directory is based on [AllenNLP](https://github.com/allenai/allennlp), which was originally released under the Apache 2.0 license.

## Getting Help

Post an issue here on GitHub if you have any problems, and create a pull request if you make any improvements (substantial or cosmetic) to the code that you're willing to share.

## FAQs

It seems like my preproc/{task}__{split}.data has nothing in it!

This probably means that you probably ran the script before downloading the data for that task. Thus, delete the file from preproc and then run main.py again to build the data splits from scratch.
