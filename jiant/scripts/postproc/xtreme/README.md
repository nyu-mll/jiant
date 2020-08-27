# XTREME Running / Submission Guide

This guide will walk through the full process for evaluating XLM-R on all [XTREME](https://sites.research.google/xtreme/) tasks. It consists of the following steps:

* [Download Model](#download-model)
* [Download Data](#download-data)
* [Tokenize and Cache Data](#tokenize-and-cache-data)
* [Generate Run Configs](#generate-run-configs)
* [Train/Run Models](#trainrun-models)

You can also choose to just run one of the tasks, instead of the whole benchmark.

This code is largely based on the [reference implementation](https://github.com/google-research/xtreme) for the XTREME benchmark.

Before we begin, be sure to set the following environment variables:
```bash
BASE_PATH=/path/to/xtreme/experiments
MODEL_TYPE=xlm-roberta-large
``` 

We will link to bash **code snippets** in the section below. We recommend that you open them side-by-side with this README when reading them. They should also all work if you run them all in order, after setting the above environment variables.

## Download Model

First, we download the model we want to us. From the `MODEL_TYPE` variable above, we are using `xlm-roberta-large`.

See: [**Code snippet**](./subscripts/a_download_model.sh)

## Download Data

Next, we download the XTREME data. We also need to download the MNLI and SQuAD datasets as training data for XNLI, XQuAD and MLQA.

See: [**Code snippet**](./subscripts/b_download_data.sh)

## Tokenize and Cache Data

Now, we preprocess our data into a tokenized cache. We need to do this across all languages for each XTREME task, as well as MNLI and SQuAD. Somewhat tediously (and this will come up again), different tasks have slightly different phases (train/val/test) available, so we have slightly different configurations for each.

See: [**Code snippet**](./subscripts/c_tokenize_and_cache.sh)

## Generate Run configs

Now, we generate the run configurations for each of our XTREME tasks. Each of the 9 XTREME tasks will correspond to one run. It's worth noting here:

* XNLI uses MNLI for training, XQuAD and MLQA use SQuAD v1.1, while PAWS-X, UDPOS, PANX and TyDiQA have their own English training sets.
* For these tasks, we will train on the training set, and then evaluate the validation set for all available languages.
* Bucc2018 and Tatoeba, the sentence retrieval tasks, are not trained, and only run in evaluation mode.
* We need to ensure that all tasks in a single run use the exact same output head. This is prepared for you in the `xtreme_runconfig_writer`. We recommend looking over the resulting run config file to verify how the run is set up.
* In theory, we could do XQuAD and MLQA in a single run, since they are both trained on SQuAD and evaluated zero-shot. For simplicity, we will treat them as separate runs. You can combine them into a single run config by modifying the runconfig file. 

See: [**Code snippet**](./subscripts/d_write_configs.sh)

## Train/Run models

Now, we can fine-tune XLM-R on each task (in the cases of Bucc2018 and Tatoeba, we just run evaluation). 

We put this in the format for a bash loop here, but we recommend running these commands in parallel, one job for each task, if you have a cluster available.

See: [**Code snippet**](./subscripts/e_run_models.sh)
