# Edge Probing

This is the main page for [What do you learn from context? Probing for sentence structure in contextualized word representations](https://openreview.net/forum?id=SJzSgnRcKX), a.k.a. "Edge Probing."

## Getting Started

First, follow the set-up instructions for `jiant`: [Getting Started](../README.md#getting-started)  
In particular, you'll need to set the following environment variables:
- `JIANT_PROJECT_PREFIX` to wherever you want experiments to be saved (like 
  `$HOME/exp`)
- `JIANT_DATA_DIR` to the directory where you'll download the edge probing data 
  (like `$HOME/jiant_data`)
- `GLOVE_EMBS_FILE` to a copy of `glove.840B.300d.txt` (get it 
  [here](http://nlp.stanford.edu/data/glove.840B.300d.zip)) if you want to use GloVe or CoVe.
- _TODO(ian): add optional cache paths for ELMo and BERT_

Next, download and process the edge probing data. You'll need access to the underlying corpora, in particular OntoNotes 5.0 and a processed (JSON) copy of the SPR1 dataset. Edit the paths in [get_and_process_all_data.sh](get_and_process_all_data.sh) to point to these resources, then run:

```sh
mkdir -p $JIANT_DATA_DIR
./get_and_process_all_data.sh $JIANT_DATA_DIR
```
This should populate `$JIANT_DATA_DIR/edges` with directories for each task, each containing a number of `.json` files as well as `labels.txt`. For more details on the data format, see below.

The main entry point for edge probing is [`jiant/main.py`](../main.py). The main arguments are a config file and any parameter overrides. The [`jiant/config/`](../config/) folder contains [HOCON](https://github.com/lightbend/config/blob/master/HOCON.md) files as a starting point for all the edge probing experiments. 

For a quick test run, use a small dataset like `spr2` and a small encoder like CoVe:
```sh
cd ${PWD%/jiant*}/jiant
python main.py --config_file config/edgeprobe_cove.conf \
  -o "target_tasks=edges-spr2,exp_name=ep_cove_demo"
```
This will keep the encoder fixed and train an edge probing classifier on the SPR2 dataset. It should run in about 4 minutes on a K80 GPU. It'll produce an output directory in `$JIANT_PROJECT_PREFIX/ep_cove_demo`. There's a lot of stuff in here, but the files of interest are:
```
vocab/
  tokens.txt             # token vocab used by the encoder
  edges-spr2_labels.txt  # label vocab used by the probing classifier
run/
  tensorboard/              # tensorboard logdir
  edges-spr2_val.json       # dev set predictions, in edge probing JSON format
  edges-spr2_test.json      # test set predictions, in edge probing JSON format
  log.log                   # training and eval log file (human-readable text)
  model_state_eval_best.th  # PyTorch saved checkpoint
  params.conf               # serialized parameter list
```
`jiant` uses [tensorboardX](https://github.com/lanpa/tensorboardX) to record loss curves and a few other metrics during training. You can view with:
```
tensorboard --logdir $JIANT_PROJECT_PREFIX/ep_cove_demo/run/tensorboard
```

You can use the `run/*_val.json` and `run/*_test.json` files to run scoring and analysis. There are some helper utilities which allow you to load and aggregate predictions across multiple runs. In particular:
- [analysis.py](analysis.py) contains utilities to load predictions into a set 
  of DataFrames, as well as to pretty-print edge probing examples.
- [edgeprobe_preds_sandbox.ipynb](edgeprobe_preds_sandbox.ipynb) walks through 
  some of the features in `analysis.py`
- [analyze_runs.py](analyze_runs.py) is a helper script to process a set of 
  predictions into a condensed `.tsv` format. It computes confusion matricies for each label and along various stratifiers (like span distance) so you can easily and quickly perform further aggregation and compute metrics like accuracy, precision, recall, and F1. In particular, the `run`, `task`, `label`, `stratifier` (optional), and `stratum_key` (optional) columns serve as identifiers, and the confusion matrix is stored in four columns: `tp_count`, `fp_count`, `tn_count`, and `tp_count`. If you want to aggregate over a group of labels (like SRL core roles), just sum the `*_count` columns for that group before computing metrics.
- [edgeprobe_aggregate_analysis.ipynb](edgeprobe_aggregate_analysis.ipynb) is 
  the notebook we used to generate the tables and figures in the paper. It's kind of messy, but ultimately just does some shallow processing over the output of `analyze_runs.py`. If you're trying to do your own analysis, the main idiosyncracies to know about are: for `ontonotes-coref`, use the `1` label instead of `_micro_avg_`, and for `srl-conll2012` we report a `_clean_micro_` metric which aggregates all the labels that don't start with `R-` or `C-`.

## Running the experiments from the paper

We provide a frozen branch, [`edgeprobe_frozen_feb2019`](https://github.com/jsalt18-sentence-repl/jiant/tree/edgeprobe_frozen_feb2019), which should reflect the master branch as of the final version of the paper.

The configs in `jiant/config/edgeprobe_*.conf` are the starting point for the experiments in the paper, but are supplemented by a number of parameter overrides (the `-o` flag to `main.py`). We use a set of bash functions to keep track of these, which are maintained in [`jiant/scripts/edges/exp_fns.sh`](../scripts/edges/exp_fns.sh).

To run a standard experiment, you can do something like:
```sh
pushd ${PWD%/jiant*}/jiant
source scripts/edges/exp_fns.sh
# Run a lexical baseline (ELMo char CNN)
elmo_chars_exp edges-srl-conll2012
```

The paper (Table 2 in particular) represents the output of a large number of experiments. Some of these are quite fast (lexical baselines and CoVe), and some are quite slow (GPT model, syntax tasks with lots of targets). We use a Kubernetes cluster running on Google Cloud Platform (GCP) to manage all of these. For more on Kubernetes, see [`jiant/gcp/kubernetes`](../gcp/kubernetes).

The master script for the experiments is [`jiant/scripts/edges/kubernetes_run_all.sh`](../scripts/edges/kubernetes_run_all.sh). Mostly, all this does is set up some paths and submit pods to run on the cluster. If you want to run the same set of experiments in a different environment, you can copy that script and modify the `kuberun()` function to submit a job or to simply run locally.

If you hit any snags (_Editor's note: it's research code, you probably will_), contact Ian (email address in the paper) for help.

## Edge Probing Utilities

This directory contains a number of utilities for the edge probing project.

In particular:
- [edge_data_stats.py](edge_data_stats.py) prints stats, like the number of 
  tokens, number of spans, and number of labels.
- [get_edge_data_labels.py](get_edge_data_labels.py) compiles a list of all the 
  unique labels found in a dataset.
- [retokenize_edge_data.py](retokenize_edge_data.py) applies tokenizers (MosesTokenizer, OpenAI.BPE, or a BERT wordpiece model) and re-map spans to the new tokenization.
- [convert_edge_data_to_tfrecord.py](convert_edge_data_to_tfrecord.py) converts 
  edge probing JSON data to TensorFlow examples.

The [data/](data/) subdirectory contains scripts to download each probing dataset and convert it to the edge probing JSON format, described below.

If you just want to get all the data, see [get_and_process_all_data.sh](get_and_process_all_data.sh); this is a convenience wrapper over the instructions in [data/README.md](data/README.md).

## Data Format

The edge probing data is stored and manipulated as JSON (or the equivalent Python dict) which encodes a single `text` field and a number of `targets` each consisting of `span1`, (optionally) `span2`, and a list of `labels`. The `info` field can be used for additional metadata. See examples below:

**SRL example**
```js
// span1 is predicate, span2 is argument
{
  “text”: “Ian ate strawberry ice cream”,
  “targets”: [
    { “span1”: [1,2], “span2”: [0,1], “label”: “A0” },
    { “span1”: [1,2], “span2”: [2,5], “label”: “A1” }
  ],
  “info”: { “source”: “PropBank”, ... }
}
```

**Constituents example**
```js
// span2 is unused
{
  “text”: “Ian ate strawberry ice cream”,
  “targets”: [
    { “span1”: [0,1], “label”: “NNP” },
    { “span1”: [1,2], “label”: “VBD” },
    ...
    { “span1”: [2,5], “label”: “NP” }
    { “span1”: [1,5], “label”: “VP” }
    { “span1”: [0,5], “label”: “S” }
  ]
  “info”: { “source”: “PTB”, ... }
}
```

**Semantic Proto-roles (SPR) example**
```js
// span1 is predicate, span2 is argument
// label is a list of attributes (multilabel)
{
  'text': "The main reason is Google is more accessible to the global community and you can rest assured that it 's not going to go away ."
  'targets': [
    {
      'span1': [3, 4], 'span2': [0, 3],
      'label': ['existed_after', 'existed_before', 'existed_during',
                'instigation', 'was_used'],
      'info': { ... }
    },
    ...
  ]
  'info': {'source': 'SPR2', ... },
}
```

## Labels and Retokenization

For each task, we need to perform two additional preprocessing steps before training using main.py.

First, extract the set of available labels:
```
export TASK_DIR="$JIANT_DATA_DIR/edges/<task>"
python jiant/probing/get_edge_data_labels.py -o $TASK_DIR/labels.txt \
    -i $TASK_DIR/*.json -s
```

Second, make retokenized versions for MosesTokenizer and for the OpenAI BPE model:
```
python jiant/probing/retokenize_edge_data.py -t "MosesTokenizer" $TASK_DIR/*.json
python jiant/probing/retokenize_edge_data.py -t "OpenAI.BPE"     $TASK_DIR/*.json
```
This will make retokenized versions alongside the original files.
