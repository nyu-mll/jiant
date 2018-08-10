# JSALT: *J*iant (or *J*SALT) *S*entence *A*ggregating *L*earning *T*hing
This repo contains the code for the jiant sentence representation learning model used at the [2018 JSALT Workshop](https://www.clsp.jhu.edu/workshops/18-workshop/) by the [General-Purpose Sentence Representation Learning](https://jsalt18-sentence-repl.github.io/) team.

## Quick-Start on GCP

If you're using Google Compute Engine, the project instance images (`cpu-workstation-template*` and `gpu-worker-template-*`) already have all the required packages installed, plus the GLUE data and pre-trained embeddings downloaded to `/usr/share/jsalt`. Clone this repo to your home directory, then test with:

```sh
python main.py --config_file config/demo.conf
```

You should see the model start training, and achieve an accuracy of > 70% on SST in a few minutes. The default config will write the experiment directory to `$HOME/exp/<experiment_name>` and the run directory to `$HOME/exp/<experiment_name>/<run_name>`, so you can find the demo output in `~/exp/jiant-demo/sst`.

## Dependencies

Make sure you have installed the packages listed in `environment.yml`.
When listed, specific particular package versions are required.
If you use conda (recommended, [instructions for installing miniconda here](https://conda.io/miniconda.html)), you can create an environment from this package with the following command:

```
conda env create -f environment.yml
```

To activate the environment run ``source activate jiant``, and to deactivate run ``source deactivate``

## Downloading data

The repo contains a convenience python script for downloading all [GLUE](https://www.nyu.edu/projects/bowman/glue.pdf) data and standard splits.

```
python download_glue_data.py --data_dir data --tasks all
```

We also make use of many other data sources, including:

- translation: WMT'14 EN-DE, WMT'17 EN-RU # TODO(Edouard,Katherin)
- language modeling: [Billion Word Benchmark](http://www.statmt.org/lm-benchmark/), [WikiText103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) # TODO(Yinghui): describe preprocessing done or make preprocessed data available?
- image captioning: # TODO(Roma)
- Reddit: # TODO(Raghu)
- DisSent: # TODO(Tom): describe preprocessing done or make preprocessed data available?
- Recast data: # TODO(Ellie?,Adam?)
- CCG: # TODO(Tom)

To incorporate the above data, placed the data in the data directory in its own directory (see task-directory relations in `src/preprocess.py` and `src/tasks.py`.

## Running

To run an experiment, make a config file similar to `config/demo.conf` with your model configuration. You can use the `--overrides` flag to override specific variables. For example:
```sh
python main.py --config_file config/demo.conf --overrides "exp_name = my_exp, run_name = foobar"
```
will run the demo config, but output to `$JIANT_PROJECT_PREFIX/my_exp/foobar`.

Because preprocessing is expensive, we often want to run multiple experiments using the same preprocessing. So, we group runs using the same preprocessing in a single experiment directory (set using the ``exp_dir`` flag) in which we store all shared preprocessing objects. Later runs will load the stored preprocessing. We write run-specific information (logs, saved models, etc.) to a run-specific directory (set using flag ``run_dir``), usually nested in the experiment directory. Overall the directory structure looks like:

```
exp1/ (e.g. training and evaluating on Task1 and Task2)
 |-- preproc/ # shared indexed data of Task1 and Task2
 |-- vocab/ # shared vocabulary built from examples from Task1 and Task 2
 |-- Task1/ # shared Task1 class object
 |-- Task2/ # shared Task2 class object
 |-- run1/ # run directory with some hyperparameter settings
 |-- run2/ # run directory with some different hyperparameter settings
 |-- [...]
exp2/ (e.g. training and evaluating on Task1 and Task3)
 |-- preproc/
 |-- vocab/ 
 |-- Task1/
 |-- Task3/
 |-- run1/
 |-- [...]
```

You should also be sure to set ``data_dir`` and  ``word_embs_file`` options to point to the directories containing the data (e.g. the output of the ``download_glue_data`` script and word embeddings (see later sections) respectively). Though, if you are using the preconfigured GCP instance templates these may already be set!

To force rereading and reloading of the tasks, perhaps because you changed the format or preprocessing of a task, use the option ``reload_tasks = 1``.

To force rebuilding of the vocabulary, perhaps because you want to include vocabulary for more tasks, use the option ``reload_vocab = 1``.

To force reindexing of a task's data, use the option ``reload_index = 1`` and set ``reindex_tasks`` to the names of the tasks to be reindexed, e.g. ``reindex_tasks="sst,mnli"``.

## Model

The core model is a shared BiLSTM with task-specific components. When a language modeling objective is included in the set of training tasks, we use a bidirectional language model for all tasks, which is constructed to avoid cheating on the language modeling tasks.

We also include an experimental, untested option to use a shared [Transformer](https://arxiv.org/abs/1706.03762) in place of the shared BiLSTM by setting ``sent_enc = transformer``. When using a Transformer, we use the [Noam learning rate scheduler](https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers.py#L84), as that seems important to training the Transformer thoroughly.

Task-specific components include logistic regression and multi-layer perceptron for classification and regression tasks, and an RNN decoder with attention for sequence transduction tasks.
To see the full set of available params, see [config/defaults.conf](config/defaults.conf) and the brief arguments section in [main.py](main.py).

## Trainer

The trainer was originally written to perform sampling-based multi-task training. At each step, a task is sampled and ``bpp_base`` (default: 1) batches of that task's training data is trained on.
The trainer evaluates the model on the validation data after a fixed number of updates, set by ``val_interval``.
The learning rate is scheduled to decay by ``lr_decay_factor`` (default: .5) whenever the validation score doesn't improve after ``task_patience`` (default: 1) validation checks.

If you're training only on one task, you don't need to worry about sampling schemes, but if you are training on multiple tasks, you can vary the sampling weights with ``weighting_method``, e.g. ``weighting_method = uniform`` or ``weighting_method = proportional`` (to amount of training data). You can also scale the losses of each minibatch via ``scaling_method`` if you want to weight tasks with different amounts of training data equally throughout training. 

Within a run, tasks are distinguished between training tasks and evaluation tasks. The logic of ``main.py`` is that the entire model is trained on all the training tasks, then the best model is loaded, and task-specific components are trained for each of the evaluation tasks with a frozen shared sentence encoder.
You can control which steps are performed or skipped by setting the flags ``do_train, train_for_eval, do_eval``.
Specify training tasks with ``train_tasks = $TRAIN_TASKS `` where ``$TRAIN_TASKS`` is a comma-separated list of task names; similarly use ``eval_tasks`` to specify the eval-only tasks.
Note: if you want to train and evaluate on a task, that task must be in both ``train_tasks`` and ``eval_tasks``.

NB: "epoch" is generally used to refer to the amount of data between validation checks.


## Adding New Tasks

To add new tasks, you should:
1. Add your data to the ``data_dir`` you intend to use. When constructing your task class (see next bullet), make sure you specify the correct subfolder containing your data. 

2. Create a class in ``src/tasks.py``, making sure that:
    - Decorate the task: in the line immediately before ``class MyNewTask():``, add the line ``@register_task(task_name, rel_path='path/to/data')`` where ``task_name`` is the dedesignation for the task used in ``train_tasks, eval_tasks`` and ``rel_path`` is the path to the data in ``data_dir``.
    - Your task inherits from existing classes as necessary (e.g. ``PairClassificationTask``, ``SequenceGenerationTask``, ``WikiTextLMTask``, etc.).
    - The task definition should include the data loader, as a method called ``load_data()`` which stores tokenized but un-indexed data for each split in attributes named ``task.{train,valid,test}_data_text``. The formatting of each datum can be anything as long as your preprocessing code (in ``src/preprocess.py``, see next bullet) expects that format. Generally data are formatted as lists of inputs and output, e.g. MNLI is formatted as ``[[sentences1]; [sentences2]; [labels]]`` where ``sentences{1,2}`` is a list of the first sentences from each example. Make sure to call your data loader in initialization!
    - Your task should implement a method ``task.get_sentences()`` that iterates over all text to index in order to build the vocabulary. For some types of tasks, e.g. ``SingleClassificationTask``, you only need set ``task.sentences`` to be a list of sentences (``List[List[str]]``).
    - Your task should implement a method ``task.count_examples()`` that sets ``task.example_counts`` (``Dict[str:int]``): the number of examples per split (train, val, test).
    - Your task should implement a method ``task.get_split_text()`` that takes in the name of a split and returns an iterable over the data in that split.
    - Your task should implement a method ``task.process_split()`` that takes in a split of your data and produces a list of AllenNLP ``Instance``s. An ``Instance`` is a wrapper around a dictionary of ``(field_name, Field)`` pairs. ``Field``s are objects to help with data processing (indexing, padding, etc.). Each input and output should be wrapped in a field of the appropriate type (``TextField`` for text, ``LabelField`` for class labels, etc.). For MNLI, we wrap the premise and hypothesis in ``TextField``s and the label in ``LabelField``. See the [AllenNLP tutorial](https://allennlp.org/tutorials) or the examples in ``src/tasks.py``.  The names of the fields, e.g. ``input1``, can be named anything so long as the corresponding code in ``src/models.py`` (see next bullet) expects that named field. However make sure that the values to be predicted are either named ``labels`` (for classification or regression) or ``targs`` (for sequence generation)!
    - If you task requires task specific label namespaces, e.g. for translation or tagging, you should set the attribute ``task._label_namespace`` to reserve a vocabulary namespace for your task's target labels. We strongly suggest including the task name in the target namespace. Your task should also implement ``task.get_all_labels()``, which returns an iterable over the labels (possibly words, e.g. in the case of MT) in the task-specific namespace.

3. In ``src/models.py``, make sure that:
    - The correct task-specific module is being created for your task in ``build_module()``.
    - Your task is correctly being handled in ``forward()`` of ``MultiTaskModel``. The model will receive the task class you created and a batch of data, where each batch is a dictionary with keys of the ``Instance`` objects you created in preprocessing, as well as a ``predict`` flag that indicates if your forward function should generate predictions or not.
    - Create additional methods or add branches to existing methods as necessary. If you do add additional methods, make sure to make use of the ``sent_encoder`` attribute of the model, which is shared amongst all tasks.
Note: The current training procedure is task-agnostic: we randomly sample a task to train on, pass a batch to the model, and receive an output dictionary at least containing a ``loss`` key. Training loss should be calculated within the model; validation metrics should also be computed within AllenNLP ``scorer``s and not in the training loop. So you should *not* need to modify the training loop; please reach out if you think you need to.

## Pretrained Embeddings

### FastText

To use fastText, we can either use the pretrained vectors or pretrained model. The former will have OOV terms while the latter will not, so using the latter is preferred.
To use the pretrained model, follow the instructions [here](https://github.com/facebookresearch/fastText) (specifically "Building fastText for Python") to setup the fastText package, then download the trained English [model](https://fasttext.cc/docs/en/pretrained-vectors.html) (note: 9.6G).
fastText will also need to be built in the jiant environment following [these instructions](https://github.com/facebookresearch/fastText#building-fasttext-for-python).
To activate fastText model within our framework, set the flag ``fastText = 1``

Download the pretrained vectors located [here](https://fasttext.cc/docs/en/english-vectors.html), preferrably the 300-dimensional Common Crawl vectors. Set the ``word_emb_file`` to point to the .vec file.

### ELMo

We use the ELMo implementation provided by [AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md).
To use ELMo, set ``elmo`` to 1.
By default, AllenNLP will download and cache the pretrained ELMo weights. If you want to use a particular file containing ELMo weights, set ``elmo_weight_file_path = path/to/file``.

To use only the _character-level CNN word encoder_ from ELMo by use `elmo_chars_only = 1` (set by default).

### GloVe

Many of our models make use of [GloVe pretrained word embeddings](https://nlp.stanford.edu/projects/glove/), in particular the 300-dimensional, 840B version.
To use GloVe vectors, download and extract the relevant files and set ``word_embs_file`` to the GloVe file.

### CoVe

We use the CoVe implementation provided [here](https://github.com/salesforce/cove).
To use CoVe, clone the repo and set the option ``path_to_cove = "/path/to/cove/repo"`` and set ``cove = 1``.

## Getting Help

Feel free to contact alexwang _at_ nyu.edu with any questions or comments.
