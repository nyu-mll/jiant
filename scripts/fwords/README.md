# Instructions
This branch contains a version of the `jiant` code used in [Probing What Different NLP Tasks Teach Machines about Function Word Comprehension](https://arxiv.org/abs/1904.11544). This directory contains scripts to run experiments described in the paper. Contact Najoung (see paper for email) if you run into any issues.

## Getting started
First, set up `jiant` following the main instructions.

Then download the function word probing datasets from [the DNC repo](https://github.com/decompositional-semantics-initiative/DNC/tree/master/function_words). Place the dataset files in the data directory (as indicated by `${JIANT_DATA_DIR}`). 

Since the acceptability datasets are evaluated using 10-fold cross-validation, we provide a corresponding `-folds.tsv` file for each acceptability dataset, indicating the examples used for train, dev, test in each fold. You can use `preprocess_acceptability_folds.py` to automatically generate the folds using the downloaded `.json` files. 

By default, the code looks for `foldn` directories (_n_=number of xval fold) under directories named `definiteness/`, `coordinating-conjunctions/`, `eos/`, and `whwords/`. You can change this in the `@register_task`decorators of each task in `src/tasks.py`.

## Running experiments from the paper

### Pretraining different models
Use `pretrain.sh`. You can modify `PRETRAIN_TASK` to change the pretraining task. Use the names of pretraining tasks as defined by `@register_task` in `src/tasks.py`. 

The names corresponding to pretraining tasks used in the paper are: `ccg`, `dissentwikifullbig`, `mnli`, `grounded`, `skipthought`, `wmt14_en_de`, `bwb`. 

### Evaluating on acceptability tasks
Use `probe_acceptability.sh`. Modify `MODEL_DIR` and `MODEL_FILE` to the directory and the model file of the pretrained model (the result of `pretrain.sh`) that you would like to probe. 

You can modify `PROBING_TASK` to change the probing task. The available options are: `acceptability-conj`, `acceptability-def`, `acceptability-eos`, `acceptability-wh`.


### Evaluating on NLI tasks
First, you need to train an MNLI classifier on top of the pretrained model (see paper for details). Use `train_nli_mlp.sh`, modifying `MODEL_DIR` and `MODEL_FILE` to point to the pretrained model.

Then use `probe_nli.sh`. Modify `MODEL_DIR` to provide the path to the trained MNLI classifier directory. 

Again, you can modify `PROBING_TASK` to change the probing task. The available options are: `nli-prob-prep`, `nli-prob-negation`, `nli-prob-spatial`, `nli-prob-quant`, `nli-prob-comp`.

### Notes
- To pretrain a model on language modeling or to evaluate a model pretrained on language modeling, you have to additionally set `sent_enc=bilm`. This is also noted in the script files. 
