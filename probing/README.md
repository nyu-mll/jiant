# Edge Probing Utils

This directory contains a number of utilities for the edge probing project.

In particular:
- [edge_data_stats.py](edge_data_stats.py) prints stats, like the number of 
  tokens, number of spans, and number of labels.
- [get_edge_data_labels.py](get_edge_data_labels.py) compiles a list of all the 
  unique labels found in a dataset.
- [retokenize_edge_data.py](retokenize_edge_data.py) and 
  [retokenize_edge_data.openai.py](retokenize_edge_data.openai.py) apply tokenizers (MosesTokenizer or the BPE model from OpenAI GPT) and re-map spans to the new tokenization.
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
python jiant/probing/retokenize_edge_data.py $TASK_DIR/*.json
python jiant/probing/retokenize_edge_data.openai.py $TASK_DIR/*.json
```
This will make retokenized versions alongside the original files.
