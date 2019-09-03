# Edge Probing

This is the main page for the following papers:

- **What do you learn from context? Probing for sentence structure in contextualized word representations** (Tenney et al., ICLR 2019), the "edge probing paper": [[paper](https://openreview.net/forum?id=SJzSgnRcKX)] [[poster](https://iftenney.github.io/edgeprobe-poster-iclr-final.pdf)]
- **BERT Rediscovers the Classical NLP Pipeline** (Tenney et al., ACL 2019), the "BERT layer paper":[[paper](https://arxiv.org/abs/1905.05950)] [[poster](https://iftenney.github.io/bert-layer-poster-acl-final.pdf)]

Most of the code for these is integrated into `jiant` proper, but this directory contains data preparation and analysis code specific to the edge probing experiments. Additionally, the runner scripts live in [jiant/scripts/edgeprobing](../scripts/edgeprobing).

## Getting Started

First, follow the set-up instructions for `jiant`: [Getting Started](../README.md#getting-started). Be sure you set all the required environment variables, and that you download the git submodules.

If you want to run GloVe or CoVe experiments, also be sure to set `WORD_EMBS_FILE` to point to a copy of [`glove.840B.300d.txt`](http://nlp.stanford.edu/data/glove.840B.300d.zip).

Next, download and process the edge probing data. You'll need access to the underlying corpora, in particular OntoNotes 5.0 and a processed (JSON) copy of the SPR1 dataset. Edit the paths in [get_and_process_all_data.sh](get_and_process_all_data.sh) to point to these resources, then run:

```sh
mkdir -p $JIANT_DATA_DIR
./get_and_process_all_data.sh $JIANT_DATA_DIR
```
This should populate `$JIANT_DATA_DIR/edges` with directories for each task, each containing a number of `.json` files as well as `labels.txt`. For more details on the data format, see below.

The main entry point for edge probing is [`jiant/main.py`](../main.py). The main arguments are a config file and any parameter overrides. The [`jiant/jiant/config/edgeprobe/`](../jiant/config/edgeprobe/) folder contains [HOCON](https://github.com/lightbend/config/blob/master/HOCON.md) files as a starting point for all the edge probing experiments.

For a quick test run, use a small dataset like `spr2` and a small encoder like CoVe:
```sh
cd ${PWD%/jiant*}/jiant
python main.py --config_file jiant/config/edgeprobe/edgeprobe_cove.conf \
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
  params.conf               # serialized parameter list
  edges-spr2/model_state_eval_*.best.th  # PyTorch saved checkpoint
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
- [get_scalar_mix.py](get_scalar_mix.py) is a helper script to extract scalar
  mixing weights and export to `.tsv`.
- [analysis_edgeprobe_standard.ipynb](analysis_edgeprobe_standard.ipynb) shows
  some example analysis on the output of `analyze_runs.py` and `get_scalar_mix.py`. This mostly does shallow processing over the output, but the main idiosyncracies to know are: for `coref-ontonotes`, use the `1` label instead of `_micro_avg_`, and for `srl-ontonotes` we report a `_clean_micro_` metric which aggregates all the labels that don't start with `R-` or `C-`.

## Running the experiments from the paper

We provide a frozen branch, [`ep_frozen_20190723`](https://github.com/nyu-mll/jiant/tree/ep_frozen_20190723), which should reproduce the experiments from both papers above.

Additionally, there's an older branch, [`edgeprobe_frozen_feb2019`](https://github.com/jsalt18-sentence-repl/jiant/tree/edgeprobe_frozen_feb2019), which is a snapshot of `jiant` as of the final version of the ICLR paper. However, this is much messier than above.

The configs in `jiant/jiant/config/edgeprobe/edgeprobe_*.conf` are the starting point for the experiments in the paper, but are supplemented by a number of parameter overrides (the `-o` flag to `main.py`). We use a set of bash functions to keep track of these, which are maintained in [`jiant/scripts/edges/exp_fns.sh`](../scripts/edges/exp_fns.sh).

To run a standard experiment, you can do something like:
```sh
pushd ${PWD%/jiant*}/jiant
source scripts/edges/exp_fns.sh
bert_mix_exp edges-srl-ontonotes bert-base-uncased
```

The paper (Table 2 in particular) represents the output of a large number of experiments. Some of these are quite fast (lexical baselines and CoVe), and some are quite slow (GPT model, syntax tasks with lots of targets). We use a Kubernetes cluster running on Google Cloud Platform (GCP) to manage all of these. For more on Kubernetes, see [`jiant/gcp/kubernetes`](../gcp/kubernetes).

The master script for the experiments is [`jiant/scripts/edgeprobing/kubernetes_run_all.sh`](../scripts/edgeprobing/kubernetes_run_all.sh). Mostly, all this does is set up some paths and submit pods to run on the cluster. If you want to run the same set of experiments in a different environment, you can copy that script and modify the `kuberun()` function to submit a job or to simply run locally.

There's also an analysis helper script, [jiant/scripts/edgeprobing/analyze_project.sh](../scripts/edgeprobing/analyze_project.sh), which runs `analyze_runs.py` and `get_scalar_mix.py` on the output of a set of Kubernetes runs. Note that scoring runs is CPU-intensive and might take a while for larger experiments.

There are two analysis notebooks which produce the main tables and figures for each paper. These are frozen as-is for a reference, but probably won't be runnable directly as they reference a number of specific data paths:
- [analysis_edgeprobe_ICLR_camera_ready.ipynb](analysis_edgeprobe_ICLR_camera_ready.ipynb)
- [analysis_bertlayer_ACL_camera_ready.ipynb](analysis_bertlayer_ACL_camera_ready.ipynb)

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

Second, make retokenized versions for any tokenizers you need. For example:
```sh
# for CoVe and GPT, respectively
python jiant/probing/retokenize_edge_data.py -t "MosesTokenizer" $TASK_DIR/*.json
python jiant/probing/retokenize_edge_data.py -t "OpenAI.BPE"     $TASK_DIR/*.json
# for BERT
python jiant/probing/retokenize_edge_data.py -t "bert-base-uncased"  $TASK_DIR/*.json
python jiant/probing/retokenize_edge_data.py -t "bert-large-uncased" $TASK_DIR/*.json
```

This will save retokenized versions alongside the original files.
