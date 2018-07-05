# JSALT: *J*iant (or *J*SALT) *S*entence *A*ggregating *L*earning *T*hing
This repo contains the code for jiant sentence representation learning model for the 2018 JSALT Workshop.

## Quick-Start on GCP

If you're using Google Compute Engine, the project instance images (`cpu-workstation-template*` and `gpu-worker-template-*`) already have all the required packages installed, plus the GLUE data and pre-trained embeddings downloaded to `/usr/share/jsalt`. Clone this repo to your home directory, then test with:

```sh
python src/main.py --config_file config/demo.conf
```

You should see the model start training, and achieve an accuracy of > 70% on SST in a few minutes. The default config will write the experiment directory to `$HOME/exp/<experiment_name>` and the run directory to `$HOME/exp/<experiment_name>/<run_name>`, so you can find the demo output in `~/exp/jiant-demo/sst`.

## Dependencies

Make sure you have installed the packages listed in environment.yml.
When listed, specific particular package versions are required.
If you use conda (recommended, [instructions for installing miniconda here](https://conda.io/miniconda.html)), you can create an environment from this package with the following command:

```
conda env create -f environment.yml
```

To activate the environment run ``source activate jiant``, and to deactivate run ``source deactivate``

## Downloading data

The repo contains a convenience python script for downloading all GLUE data and standard splits.

```
python download_glue_data.py --data_dir glue_data --tasks all
```

For other pretraining task data, contact the person in charge.

## Running

To run an experiment, make a config file similar to `config/demo.conf` with your model configuration. You can use the `--overrides` flag to override specific variables. For example:
```sh
python src/main.py --config_file config/demo.conf \
  --overrides "exp_name = my_exp, run_name = foobar"
```
will run the demo config, but output to `$JIANT_PROJECT_PREFIX/my_exp/foobar`.

Because preprocessing is expensive, we often want to run multiple experiments using the same preprocessing. So, we group runs using the same preprocessing in a single experiment directory (set using the ``exp_dir`` flag) and we write run-specific information (logs, saved models, etc.) to a run-specific directory (set using flag ``run_dir``, usually nested in the experiment directory. Overall the directory structure looks like:

- exp1 (e.g. training and evaluating on WikiText and all the GLUE tasks)
    - run1 (with some hyperparameter settings)
    - run2 (with possibly the same hyperparameter settings but a different random seed)
    - run3 (with different hyperparameter settings)
- exp2 (e.g. training and evaluating on WMT and all the GLUE tasks)
    - [...]

You should also be sure to set ``data_dir`` and  ``word_embs_file`` options to point to the directories containing the data (e.g. the output of the ``download_glue_data`` script and word embeddings (see later sections) respectively). (Although note that on GCP these may already be set!)

To force rereading and reloading of the tasks, perhaps because you changed the format or preprocessing of a task, use the option ``reload_tasks = 1``.
To force rebuilding of the vocabulary, perhaps because you want to include vocabulary for more tasks, use the option ``reload_vocab = 1``.

## Model

To see the set of available params, see [config/defaults.conf](config/defaults.conf) and the brief arguments section in [src/main.py](src/main.py).


## Trainer

The trainer was originally written to perform sampling-based multi-task training. At each step, a task is sampled and one batch (to vary the number of batches to train on per sampled task, use the ``bpp_base`` of that task's training data is trained on.
The trainer evaluates the model on the validation data after a fixed number of updates, set by (``val_interval``).
The learning rate is scheduled to decay by ``lr_decay_factor`` (default: .5) whenever the validation score doesn't improve after ``task_patience`` (default: 1) validation checks.

If you're training only on one task, you don't need to worry about sampling schemes, but if you are training on multiple tasks, you can vary the sampling weights with ``weighting_method``, with options either ``uniform`` or ``proportional`` (to amount of training data). You can also scale the losses of each minibatch via ``scaling_method`` if you want to weight tasks with different amounts of training data equally throughout training.

Within a run, tasks are distinguished between training tasks and evaluation tasks. The logic of ``main.py`` is that the entire model is trained on all the training tasks, then the best model is loaded, and task-specific components are trained for each of the evaluation tasks. Specify training tasks with ``train_tasks = $TRAIN_TASKS `` where ``$TRAIN_TASKS`` is a comma-separated list of task names; similarly use ``eval_tasks`` to specify the eval-only tasks.

Other training options include:

    - ``optimizer``: (string) anything supported by AllenNLP, but usually just 'adam'
    - ``lr``: (float) set initial learning rate
    - ``batch_size``: (int) batch size, usually you want to use the largest possible, which will likely be 64 or 32 for the full model
    - ``should_train``: set to 0 to skip training
    - ``load_model``: set to 1 to start training by loading model from most recent checkpoint found in directory
    - ``force_load_epoch``: (int) after training, force loading from instead of the best epoch found during training (or the most recent if training). Useful if you have a trained model already and just want to evaluate.

NB: "epoch" is generally used to refer to the amount of data between validation checks.


## Adding New Tasks

To add new tasks, you should:
1. Add your data in a subfolder in whatever folder contains all your data ``$JIANT_DATA_DIR``. Make sure to add the correct path to the dictionary ``NAME2INFO``, structured ``task_name: (task_class, data_subdirectory)``, at the top of ``preprocess.py``. The ``task_name`` will be the commandline shortcut to train on that task, so keep it short.

2. Create a class in ``src/tasks.py``, making sure that:
    - Your task inherits from existing classes as necessary (e.g. ``PairClassificationTask``, ``SequenceGenerationTask``, etc.).
    - The task definition should include the data loader, as a method called ``load_data()`` which stores tokenized but un-indexed data for each split in attributes named ``task.{train,valid,test}_data_text``. The formatting of each datum can be anything as long as your preprocessing code (in ``src/preprocess.py``, see next bullet) expects that format. Generally data are formatted as lists of inputs and output, e.g. MNLI is formatted as ``[[sentences1]; [sentences2]; [labels]]`` where ``sentences{1,2}`` is a list of the first sentences from each example. Make sure to call your data loader in initialization!
    - Your task should include an attributes ``task.sentences`` that is a list of all text to index, e.g. for MNLI we have ``self.sentences = self.train_data_text[0] + self.train_data_text[1] + self.val_data_text[0] + ...``. Make sure to set this attribute after calling the data loader!

3. In ``src/tasks.py``, make sure that:
    - The correct task-specific preprocessing is being used for your task in ``Task.process_split()``. This should be a function that takes in a split of your data and produces a list of AllenNLP ``Instance``s. An ``Instance`` is a wrapper around a dictionary of ``(field_name, Field)`` pairs.
    - ``Field``s are objects to help with data processing (indexing, padding, etc.). Each input and output should be wrapped in a field of the appropriate type (``TextField`` for text, ``LabelField`` for class labels, etc.). For MNLI, we wrap the premise and hypothesis in ``TextField``s and the label in ``LabelField``. See the [AllenNLP tutorial](https://allennlp.org/tutorials) or the examples at the bottom of ``src/preprocess.py``.
    - The names of the fields, e.g. ``input1``, can be named anything so long as the corresponding code in ``src/model.py`` (see next bullet) expects that named field. However make sure that the values to be predicted are either named ``labels`` (for classification or regression) or ``targs`` (for sequence generation)!

4. In ``src/model.py``, make sure that:
    - The correct task-specific module is being created for your task in ``build_module()``.
    - Your task is correctly being handled in ``forward()`` of ``MultiTaskModel``. The model will receive the task class you created and a batch of data, where each batch is a dictionary with keys of the ``Instance`` objects you created in preprocessing.
    - Create additional methods or add branches to existing methods as necessary. If you do add additional methods, make sure to make use of the ``sent_encoder`` attribute of the model, which is shared amongst all tasks.
Note: The current training procedure is task-agnostic: we randomly sample a task to train on, pass a batch to the model, and receive an output dictionary at least containing a ``loss`` key. Training loss should be calculated within the model; validation metrics should also be computed within AllenNLP ``scorer``s and not in the training loop. So you should *not* need to modify the training loop; please reach out if you think you need to.

## Pretrained Embeddings

### FastText

To use fastText, we can either use the pretrained vectors or pretrained model. The former will have OOV terms while the latter will not, so using the latter is preferred.
To use the pretrained model, follow the instructions [here](https://github.com/facebookresearch/fastText) (specifically "Building fastText for Python") to setup the fastText package, then download the trained English [model](https://fasttext.cc/docs/en/pretrained-vectors.html) (note: 9.6G).
fastText will also need to be built in the jiant environment following [these instructions](https://github.com/facebookresearch/fastText#building-fasttext-for-python).
To activate fastText model within our framework, set the flag ``fastText 1``
If you get a segmentation fault running PyTorch and fastText (Sam, Alex), don't panic; use the pretrained vectors.

Download the pretrained vectors located [here](https://fasttext.cc/docs/en/english-vectors.html), preferrably the 300-dimensional Common Crawl vectors. Set the ``word_emb_file`` to point to the .vec file.

### ELMo

We use the ELMo implementation provided by [AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md).
To use ELMo, set ``elmo`` to 1.
<!-- To use ELMo without GloVe, additionally set ``elmo_no_glove`` to 1. -->

### GloVe

Many of our models make use of [GloVe pretrained word embeddings](https://nlp.stanford.edu/projects/glove/), in particular the 300-dimensional, 840B version.
To use GloVe vectors, download and extract the relevant files and set ``word_embs_file`` to the GloVe file.

### CoVe

We use the CoVe implementation provided [here](https://github.com/salesforce/cove).
To use CoVe, clone the repo and set the option ``path_to_cove = "/path/to/cove/repo"`` and set ``cove`` to 1.

## Annoying AllenNLP Things

To turn off the verbosity, you'll need to go in your AllenNLP location and create or turn on a ``quiet`` option, e.g. in ``allennlp/common/params.py``, line 186 set ``quiet=True``.
Other common and verbose locations include ``allennlp/nn/initializers.py`` (many calls to ``logger``) and ``allennlp/common/params.py`` (`pop()`` will print param values often).

To avoid needing to reconstruct vocabulary switching from using character embeddings <> not using character embeddings, using ELMo <> not using ELMo, [TODO]

## Getting Help

Feel free to contact alexwang _at_ nyu.edu with any questions or comments.
