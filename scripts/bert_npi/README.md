# Instructions
This branch contains a version of the `jiant` code used in [Investigating BERT's Knowledge of Language: Five Analysis Methods with NPIs](https://arxiv.org/abs/1909.02597). This directory contains scripts to run experiments described in the paper. Contact Haokun (hl3236 at nyu.edu) if you run into any issues.


## Set up jiant 
Follow the [tutorial](https://github.com/nyu-mll/jiant/blob/master/tutorials/setup_tutorial.md) to set up `jiant`. 

## Download the dataset
The NPI dataset, together with the code to generate it, is available [here](https://github.com/alexwarstadt/data_generation). But you can also directly download the [packed dataset](https://drive.google.com/open?id=1qoZNV7BbWbeb2YKVHpnLqomMEHFM3ZAK). Move the data into your `$JIANT_DATA_DIR`.

## Running experiments
To run the experiments, use the scripts under `jiant/scripts/bert_npi/`

### Train and evaluate models on the NPI dataset
`npi_x.sh` trains and evaluates plain BERT model and Glove BoW model on NPI dataset.
`npi_x_mnli.sh` trains and evaluates BERT --> MNLI model on NPI dataset.
`npi_x_ccg.sh` trains and evaluates BERT --> CCG model on NPI dataset.

### Train and evaluate probing classifier
`metadata_probing_exps.sh` trains and evaluates probing classifiers on NPI presence, licensor presence, and whether the NPI is in the scope
of the licensor.

### Evaluate trained models on minimal pairs
`npipair_bertnone.sh` evaluates plain BERT model on NPI minimal pairs.
`npipair_bertmnli.sh` evaluates BERT --> MNLI model on NPI minimal pairs.
`npipair_bertccg.sh` evaluates BERT --> CCG model on NPI minimal pairs.
`npipair_bow_glovenone.sh` evaluates Glove BoW model on NPI minimal pairs.

### Evaluate pretrain BERT on cloze pairs
`npimlm_bertmlm.sh` evaluates pretrained BERT model on NPI cloze pairs

## Notes
The results in our paper is averaged from 5 runs with different random seeds. You can use the `random_seed` option in the config file for it. For example, to set random seed as 42, add `random_seed = 42` to `jiant/config/bert_npi/bert.conf` and `jiant/config/bert_npi/bow_glove.conf`.
