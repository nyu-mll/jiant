# Tutorial: Getting Started


Wecome to `jiant`! Let's help get you set up and running a demo experiment!

## 1. Install

First off, let's make sure you've the full repository, including all the git submodules.

This project uses submodules to manage some dependencies on other research code, in particular for loading CoVe, GPT, and BERT. To make sure you get these repos when you download `jiant`, add `--recursive` to your `clone` command:

```
git clone --branch v0.9.1  --recursive https://github.com/nyu-mll/jiant.git jiant
```
This will download the full repository and load the 0.9 release of `jiant`. For the latest version, delete `--branch v0.9.1`. If you already cloned and just need to get the submodules, you can run:

```
git submodule update --init --recursive
```

Now, let's get your environment set up. Make sure you have `conda` installed (you can find a tutorial [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)), then run:

```
conda env create -f environment.yml
```

Make sure to activate the environment by running:

```
conda activate jiant
```

before running any `jiant` code. (To deactivate run: `source deactivate`)

Some requirements may only be needed for specific configurations. If you have trouble installing a specific dependency and suspect that it isn't needed for your use case, create an issue or a pull request, and we'll help you get by without it.

You will also need to install dependencies for `nltk` if you do not already have them:

```
python -m nltk.downloader -d  perluniprops nonbreaking_prefixes punkt
```


 ## 2. Getting data and setting up our environment

 In this tutorial, we will be working with GLUE data.
The repo contains a convenience Python script for downloading all [GLUE](https://gluebenchmark.com/tasks) data:


```
python scripts/download_glue_data.py --data_dir data --tasks all
```

We also support quite a few other data sources (check [here](https://jiant.info/documentation#/?id=data-sources)  for a list).


Finally, you'll need to set a few environment variables in [user_config_template.sh](https://github.com/nyu-mll/jiant/blob/master/user_config_template.sh), which include:


* `$JIANT_PROJECT_PREFIX`: the directory where things like logs and model checkpoints will be saved.
* `$JIANT_DATA_DIR`: location of the data you want to train and evaluate on. As a starting point, this is often the directory created by the GLUE or SuperGLUE data downloaders. Let's use the `data/` directory for GLUE for now.
* `$WORD_EMBS_FILE`: location of any word embeddings you want to use (not necessary when using ELMo, GPT, or BERT). You can download GloVe (840B) [here](http://nlp.stanford.edu/data/glove.840B.300d.zip) or fastText (2M) [here](https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip).


To avoid having your custom paths overwritten by future updates, you should save a copy of this file as `user_config.sh` (or something similar, but a file with the name `user_config.sh` will be automatically ignored by git).
Before running any experiments, you should run:

```
source user_config.sh
```

To remember to do this, it can help to run commands like this by default:

```
source user_config.sh; python main.py ...
```

Or, for a more permanent solution, run

```
source scripts/export_from_bash.sh
```

from the root of the `jiant` directory, which adds the source command directly to your machine's bash file.

Now that we've set up the environment, let's get started!

## 3. Running our first experiment

### 3.a) Configuring our experiment

Here, we'll try pretraining in a multitask setting on SST and MRPC and then finetuning on STS-B and WNLI separately using a BiLSTM sentence encoder and word embeddings trained from scratch.
This is almost exactly what is specified in `config/demo.conf`, with one major change. From here, we suggest you to go to [`config/demo.conf`](https://github.com/nyu-mll/jiant/blob/master/config/demo.conf), make a copy called `config/tutorial.conf`, and follow along - we'll explain everything that is in the file in a bit.

```
cp config/demo.conf config/tutorial.conf
```

Next, we need to make a configuration file that defines the parameters of our experiment. `config/defaults.conf` has all the documentation on the various parameters (including the ones explained below). Any config file you create should import from `config/defaults.conf`, which you can do by putting the below at the top of your config file.

```
include "defaults.conf"
```

Some important options include:

* `sent_enc`: If you want to train a new sentence encoder (rather than using a loaded one like BERT), specify it here. This is the only part of the `config/demo.conf` that we should change for our experiment since we want to train a biLSTM encoder. Thus, in your `config/tutorial.conf`, set  `sent_enc=rnn`.
* `pretrain_tasks`: This is a comma-delimited string of tasks. In `config/demo.conf`, this is set to "sst,mrpc", which is what we want. Note that we have `pretrain_tasks` as a separate field from `target_tasks` because our training loop handles the two phases differently (for example, multitask training is only supported in pretraining stage). Note that there should not be a space in-between tasks.
* `target_tasks`: This is a comma-delimited string of tasks you want to fine-tune and evaluate on (in this case "sts-b,wnli").
* `word_embs`: This is a string specifying the type of word embedding you want to use. In `config/demo.conf`, this is already set to `scratch`. *
* `val_interval`: This is the interval (in steps) at which you want to evaluate your model on the validation set during pretraining. A step is a batch update.
* `exp_name`, which expects a string of your experiment name.
* `run_name`, which expects a string of your run name.
The main differences between an experiment and a run is that all runs in an experiment will have the same preprocessing (which may include tokenization differences), while runs his a specific experiment (which may differ from other runs by learning rate or type of sentence encoder, for example, your run name could be `rnn_encoder` for a run with a biLSTM sentence encoder).

Additionally, let's suppose we want to have parameters specific to the STS-B task. Then, we can simply specify them as here:

```python

sts-b += {
    classifier_hid_dim = 512
    pair_attn = 0
    max_vals = 16
    val_interval = 10
}

```

Your finished product should look like the below:

```
include "defaults.conf"  // relative path to this file

// write to local storage by default for this demo
exp_name = jiant-demo
run_name = mtl-sst-mrpc

cuda = 0
random_seed = 42

load_model = 0
reload_tasks = 0
reload_indexing = 0
reload_vocab = 0

pretrain_tasks = "sst,mrpc"
target_tasks = "sts-b,wnli"
classifier = mlp
classifier_hid_dim = 32
max_seq_len = 10
max_word_v_size = 1000

word_embs = scratch
d_word = 50

sent_enc = rnn
skip_embs = 0

batch_size = 8

val_interval = 50
max_vals = 10
target_train_val_interval = 10
target_train_max_vals = 10

// Use += to inherit from any previously-defined task tuples.
sts-b += {
    classifier_hid_dim = 512
    pair_attn = 0
    max_vals = 16
    val_interval = 10
}

```

Now we get on to the actual experiment running!
To run the experiment, you can simply run the below command via command line, or use a bash script.

 You can use the `--overrides` flag to override specific variables. For example:

```sh
python main.py --config_file config/tutorial.conf \
    --overrides "exp_name = my_exp, run_name = foobar"
```

will run the demo config, but write output to `$JIANT_PROJECT_PREFIX/my_exp/foobar`.


### 3.b) Understanding the output logs

We support Tensorboard, however, you can also look at the logs to make sure everything in your experiment is running smoothly.

There's a lot going on here (including some debugging information that we're working on suppressing), but a lot of it is useful. The logs include:

* The process of setting up and loading tasks
* Restoring checkpoints (if applicable)
* Indexing your data into AllenNLP instances for preparation for training.
* Printing out the model architecture
* When you get to training stage, updates of the current progress and validation.

One important thing to notice is that during training, the updates will swap between sst and mrpc. This is because for each training batch, we sample the tasks based on a parameter you can set in your experiment `weighting_method`, which is automatically set to proportional (so the larger the task, the larger the probablty it will get sampled for a batch).

After validating, you will see something like this:
```
05/03 09:54:26 AM: Best result seen so far for sst.
05/03 09:54:26 AM: Best result seen so far for micro.
05/03 09:54:26 AM: Best result seen so far for macro.
05/03 09:54:26 AM: Updating LR scheduler:
05/03 09:54:26 AM:  Best result seen so far for macro_avg: 0.461
05/03 09:54:26 AM:  # epochs without improvement: 0
05/03 09:54:26 AM: mrpc_loss: training: 0.519646 validation: 1.319582
05/03 09:54:26 AM: sst_loss: training: 0.716894 validation: 0.686724
05/03 09:54:26 AM: macro_avg: validation: 0.460515
05/03 09:54:26 AM: micro_avg: validation: 0.373241
05/03 09:54:26 AM: mrpc_acc_f1: training: 0.704310 validation: 0.748025
05/03 09:54:26 AM: mrpc_accuracy: training: 0.650000 validation: 0.683824
                .
                .
                .
05/03 09:54:26 AM: sst_accuracy: training: 0.494444 validation: 0.547018

```

There are two sets of losses and scores outputted for the two tasks. This is because we're doing multitask learning in this phase.

Then, after the pretraining phase, during target task training, you will see updates for only one task at a time, and after each validation, only scores for that one task.

Lastly, we will evaluate on the target tasks, and write the results for test in your run directory.
You should see something like this:

```
05/03 09:59:15 AM: Evaluating...
05/03 09:59:15 AM: Evaluating on: sts-b, split: val
05/03 09:59:23 AM: Task 'sts-b': sorting predictions by 'idx'
05/03 09:59:23 AM: Finished evaluating on: sts-b
05/03 09:59:23 AM: Evaluating on: wnli, split: val
05/03 09:59:23 AM: Task 'wnli': sorting predictions by 'idx'
05/03 09:59:23 AM: Finished evaluating on: wnli
05/03 09:59:23 AM: Writing results for split 'val' to yo_try/jiant-demo/results.tsv
05/03 09:59:23 AM: micro_avg: 0.168, macro_avg: 0.356, sts-b_corr: 0.149, sts-b_pearsonr: 0.146, sts-b_spearmanr: 0.152, wnli_accuracy: 0.563
05/03 09:59:23 AM: Done!
```

After running this experiment, you should have in your run directory:

* a checkpoint of the best model state (based on your scores)
* a `log.log` file which contains all the logs
* a directory for each of the target_tasks containing the checkpoints of the model, task, and training state of the finetuned BERT models for that task.
* `params.conf` (a saved version of the parameters used)
* written predictions for test for each of the target trained tasks (with file names `{task_name}-test.tsv`)
* a saved checkpoint of your best validation metric.

Additionally, the validation scores will be written in `results.tsv` in your experiment directory with the name of the run it belongs to.

And there you have it! Your first experiment.
If you are looking for where to go next, check out our documentation [here](https://jiant.info/documentation/#/)!

Happy `jiant`ing!
