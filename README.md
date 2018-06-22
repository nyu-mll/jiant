# JSALT: *J*iant (or *J*SALT) *S*entence *A*ggregating *L*earning *T*hing
This repo contains the code for jiant sentence representation learning model for the 2018 JSALT Workshop.

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

To run things, use ``src/main.py`` with flags or a script like the one in ``example_experiment_scripts/demo.sh``.
Because preprocessing is expensive (particularly for ELMo), we often want to run multiple experiments using the same preprocessing. So, we group runs using the same preprocessing in a single experiment directory (set using the ``exp_dir`` flag) and we write run-specific information (logs, saved models, etc.) to a run-specific directory (set using flag ``run_dir``, usually nested in the experiment directory. Overall the directory structure looks like:

- exp1 (e.g. training and evaluating on WikiText and all the GLUE tasks)
    - run1 (with some hyperparameter settings)
    - run2 (with possibly the same hyperparameter settings)
- exp2 (e.g. training and evaluating on WMT and all the GLUE tasks)
    - [...]

You should also be sure to set ``--data_dir`` and  ``--word_embs_file`` to point to the directories containing the data (e.g. the output of the ``download_glue_data`` script and word embeddings (see later sections) respectively).

To force rereading and reloading of the tasks, perhaps because you changed the format or preprocessing of a task, use the flag ``--reload_tasks 1``.
To force rebuilding of the vocabulary, perhaps because you want to include vocabulary for more tasks, use the flag ``--reload_vocab 1``.

If you are using the experiment scripts, you should also put a file ``user_config.sh`` in the top level directory containing paths specific to your machine.

```
python main.py --data_dir $DATA_DIR --exp_dir $EXP_DIR --run_dir $RUN_DIR --train_tasks all --word_embs_file $PATH_TO_VECS
```

To use the shell script, run

```
./run_stuff.sh -d $DATA_DIR -n $EXP_DIR -r $RUN_DIR -T tasks -w $PATH_TO_VECS
```

See ``main.py`` or ``run_stuff.sh`` for options and shortcuts. A shell script was originally needed to submit to a job manager.

## Adding New Tasks

To add new tasks, you should:
1. Add your data in a subfolder in whatever folder contains all your data ``$DATA_DIR``. Make sure to add the correct path to the dictionary ``NAME2INFO``, structured ``task_name: (task_class, data_subdirectory)``, at the top of ``preprocess.py``. The ``task_name`` will be the commandline shortcut to train on that task, so keep it short.

2. Create a class in ``src/tasks.py``, making sure that:
    - Your task inherits from existing classes as necessary (e.g. ``PairClassificationTask``, ``SequenceGenerationTask``, etc.).
    - The task definition should include the data loader, as a method called ``load_data()`` which stores tokenized but un-indexed data for each split in attributes named ``task.{train,valid,test}_data_text``. The formatting of each datum can be anything as long as your preprocessing code (in ``src/preprocess.py``, see next bullet) expects that format. Generally data are formatted as lists of inputs and output, e.g. MNLI is formatted as ``[[sentences1]; [sentences2]; [labels]]`` where ``sentences{1,2}`` is a list of the first sentences from each example. Make sure to call your data loader in initialization!
    - Your task should include an attributes ``task.sentences`` that is a list of all text to index, e.g. for MNLI we have ``self.sentences = self.train_data_text[0] + self.train_data_text[1] + self.val_data_text[0] + ...``. Make sure to set this attribute after calling the data loader!

3. In ``src/preprocess.py``, make sure that:
    - The correct task-specific preprocessing is being used for your task in ``process_task()``. The recommended approach is to create a function that takes in a split of your data and produces a list of AllenNLP ``Instance``s. An ``Instance`` is a wrapper around a dictionary of ``(field_name, Field)`` pairs.
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
To activate fastText model within our framework, set the flag ``--fastText 1``
If you get a segmentation fault running PyTorch and fastText (Sam, Alex), don't panic; use the pretrained vectors.

Download the pretrained vectors located [here](https://fasttext.cc/docs/en/english-vectors.html), preferrably the 300-dimensional Common Crawl vectors. Set the ``word_emb_file`` to point to the .vec file.

### ELMo

We use the ELMo implementation provided by [AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md).
To use ELMo, set ``--elmo`` to 1. To use ELMo without GloVe, additionally set ``--elmo_no_glove`` to 1.

### GloVe

Many of our models make use of [GloVe pretrained word embeddings](https://nlp.stanford.edu/projects/glove/), in particular the 300-dimensional, 840B version.
To use GloVe vectors, download and extract the relevant files and set ``word_embs_file`` to the GloVe file.
To learn embeddings from scratch, set ``--glove`` to 0.

### CoVe

We use the CoVe implementation provided [here](https://github.com/salesforce/cove).
To use CoVe, clone the repo and fill in ``PATH_TO_COVE`` in ``src/models.py`` and set ``--cove`` to 1.

## Getting Help

Feel free to contact alexwang _at_ nyu.edu with any questions or comments.
