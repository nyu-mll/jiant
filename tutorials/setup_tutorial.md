# Tutorial: Getting Started 


Wecome to `jiant`! Let's help get you set up and running a demo experiment!


First off, let's make sure you've the full repository, including all the git submodules.

This project uses submodules to manage some dependencies on other research code, in particular for loading CoVe, GPT, and BERT. To make sure you get these repos when you download `jiant`, add `--recursive` to your `clone` command:

```
git clone --recursive git@github.com:jsalt18-sentence-repl/jiant.git jiant
```

If you already cloned and just need to get the submodules, you can run:

```
git submodule update --init --recursive
```

Now, let's get your environment set up. Make sure you have `conda` installed (you can find a tutorial [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)), then run:

```
conda env create -f environment.yml
```

Make sure to activate the environment by running 

```
conda activate jiant
``` 

before running any `jiant` code. (To deactivate run: `source deactivate`)

Some requirements may only be needed for specific configurations. If you have trouble installing a specific dependency and suspect that it isn't needed for your use case, create an issue or a pull request, and we'll help you get by without it.

You will also need to install dependencies for `nltk` if you do not already have them:

```
python -m nltk.downloader -d  perluniprops nonbreaking_prefixes punkt
``` 
***SB: This still fills my jiant directory with junk... not ideal. Either look into the suggested nltk behavior (or as a very, very hacky marginal workaround, add this to the shared .gitignore).***

This will download perluniprops in your jiant directory so don't forget to add it to your `.gitignore`!

Finally, you'll need to set a few environment variables in [user_config_template.sh](https://github.com/nyu-mll/jiant/blob/master/user_config_template.sh), which include:


* $JIANT_PROJECT_PREFIX: the directory where things like logs and model checkpoints will be saved.
* $JIANT_DATA_DIR: location of the data you want to train and evaluate on. As a starting point, this is often the directory created by the GLUE or SuperGLUE data downloaders. Let's use the `data/` directory for GLUE for now. ***SB: We haven't asked them to download any data yet. Move that earlier.***
* $WORD_EMBS_FILE: location of any word embeddings you want to use (not necessary when using ELMo, GPT, or BERT). You can download GloVe (840B) here or fastText (2M) here. ***SB: Link?***


To avoid having your custom paths overwritten by future updates, you should save a copy of this file as `user_config.sh` (or something similar, but `user_config.sh` will be automatically ignored by git.
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


Here, we'll try pretraining in a multitask setting on SST and MRPC and then finetuning on STS-B and WNLI separately using a BiLSTM sentence encoder and word embeddings trained from scratch. 
This is almost exactly what is specified in `config/demo.conf`, with one major change. From here, we suggest you to go to [`config/demo.conf`](https://github.com/nyu-mll/jiant/blob/master/config/demo.conf), make a copy in config/tutorial.conf, and follow along - we'll explain everything that is in that file in a bit. 

First, we need to download the data. We already support quite a few data sources (check [here](https://jiant.info/documentation#/?id=data-sources)  for a list). For this instance, we will need to download the GLUE data. 


The repo contains a convenience Python script for downloading all [GLUE](https://gluebenchmark.com/) data:

```
python scripts/download_glue_data.py --data_dir data --tasks all
```


Next, we need to make a configuration file that defines the parameters of our experiment. `config/defaults.conf` has a lot of documentation on the various parameters. Any config file you create should import from `config/defaults.conf`, which you can do by putting the below at the top of your config file. 

```
include "defaults.conf" 
```

Some  important options include:

* `sent_enc`: If you want to train a new sentence encoder (rather than using a loaded one like BERT), specify it here. This is the only part of the `config/demo.conf` that we should change for our experiment since we want to train a biLSTM encoder. Thus, in your `config/tutorial.conf`, set  `sent_enc=rnn`.
* `pretrain_tasks`: This is a comma-delimited string of tasks. In `config/demo.conf`, this is set to "sst,mrpc", which is what we want. Note that we have `pretrain_tasks` as a separate field from `target_tasks` because our training loop handles the two phases differently (for example, multitask training is only supported in pretraining stage).
* `target_tasks`: This is a comma-delimited string of tasks you want to fine-tune and evaluate on (in this case "sts-b,wnli").
* `word_embs`: This is a string specifying the type of word embedding you want to use. In `config/demo.conf`, this is already set to `scratch`. 
* `val_interval`: This is the interval (in steps) at which you want to evaluate your model on the validation set during pretraining.
* `exp_name`, which expects a string of your experiment name. 
* `run_name`, which expects a string of your run name.
The main differences between an experiment and a run is that all runs in an experiment will have the same preprocessing, while runs his a specific experiment (which may differ from other runs by learning rate or type of sentence encoder, for example, your run name could be `rnn_encoder` for a run with a biLSTM sentence encoder).

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


Now, something you might wonder about is what the many logs you get mean. The logs include:
* The process of setting up and loading tasks
* Restoring checkpoints (if applicable)
* Printing out the Model architecture
* Indexing your data into AllenNLP instances for preparation for training.
* When you get to training stage, updates of the current progress and validation. 

One important thing to notice is that during training, the updates will swap between sst and mrpc. This is because for each training batch, we sample the tasks to sample on. 

After validating, you will see something like this:
```
05/02 10:00:45 PM: 	Best macro_avg: 0.447
05/02 10:00:45 PM: 	# bad epochs: 0
05/02 10:00:45 PM: Statistic: mrpc_loss
05/02 10:00:45 PM: 	training: 0.577184
05/02 10:00:45 PM: 	validation: 1.190689
05/02 10:00:45 PM: Statistic: sst_loss
05/02 10:00:45 PM: 	training: 0.715676
05/02 10:00:45 PM: 	validation: 0.695592
05/02 10:00:45 PM: Statistic: macro_avg
05/02 10:00:45 PM: 	validation: 0.447327
05/02 10:00:45 PM: Statistic: micro_avg
05/02 10:00:45 PM: 	validation: 0.355272
05/02 10:00:45 PM: Statistic: mrpc_acc_f1
05/02 10:00:45 PM: 	training: 0.704310
05/02 10:00:45 PM: 	validation: 0.748025
05/02 10:00:45 PM: Statistic: mrpc_accuracy
05/02 10:00:45 PM: 	training: 0.650000
05/02 10:00:45 PM: 	validation: 0.683824
				.
                .
                .
05/02 10:00:45 PM: Statistic: sst_accuracy
05/02 10:00:45 PM: 	training: 0.519444
05/02 10:00:45 PM: 	validation: 0.520642

```

There are two sets of losses and scores outputted for the two tasks. This is because we're doing multitask learning in this phase.

Then, after the pretraining phase, during target task training, you will see updates for only one task at a time, and after each validation, only scores for that one task.

Lastly, we will evaluate on the target tasks, and write the results for test in your run directory. 
You should see something like this:

```
04/24 06:06:38 PM: Evaluating...
04/24 06:06:38 PM: Evaluating on: sts-b, split: val
04/24 06:07:08 PM: 	Task sts-b: batch 164
04/24 06:07:12 PM: Task 'sts-b': sorting predictions by 'idx'
04/24 06:07:12 PM: Finished evaluating on: sts-b
04/24 06:07:12 PM: Evaluating on: wnli, split: val
04/24 06:07:14 PM: Task 'wnli': sorting predictions by 'idx'
04/24 06:07:14 PM: Finished evaluating on: wnli
04/24 06:07:14 PM: Writing results for split 'val' to coreference_exp/jiant-demo/results.tsv
04/24 06:07:14 PM: micro_avg: 0.680, macro_avg: 0.624, sts-b_corr: 0.685, sts-b_pearsonr: 0.683, sts-b_spearmanr: 0.688, wnli_accuracy: 0.563
04/24 06:07:14 PM: Done!
```

After running this experiment, you should have in your run directory:

* a checkpoint of the best model state (based on your scores)
* a `log.log` file which contains all the logs
* `params.conf` (a saved version of the parameters used)
* written predictions for test for each of the target trained tasks (with file names {task_name}-test.tsv
* a saved checkpoint of your best validation metric.

Additionally, the validation scores will be written in `results.tsv` in your experiment directory with the name of the run it belongs to. 

And there you have it! Your first experiment.
If you are looking for where to go next, check out our documentation [here](https://jiant.info/documentation/#/?id=saving-preprocessed-data)!

Happy jianting!