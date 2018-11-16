# Edge Probing Datasets

This directory contains scripts to process the source datasets and generate the JSON data for edge probing tasks. Let `$JIANT_DATA_DIR` be the jiant data directory from the [main README](../../README.md), and make a subdirectory for the edge probing data:

```
mkdir $JIANT_DATA_DIR/edges
```

The resulting JSON has one example per line, with the following structure (line breaks added for clarity).

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

For each of the tasks below, we need to perform two more preprocessing steps.

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

## OntoNotes
Tasks:
- Constituents / POS: `edges-constituent-ontonotes`, `edges-nonterminal-ontonotes`,
  `edges-pos-ontonotes`
- Entities: `edges-ner-ontonotes`
- SRL: `edges-srl-ontonotes`
- Coreference: `edges-coref-ontonotes-conll`

### Getting OntoNotes Data
Follow the instructions at http://cemantix.org/data/ontonotes.html; you should end up with a folder named `conll-formatted-ontonotes-5.0/`.

If you're working on the JSALT cloud project, you can also download this directly from `gs://jsalt-data/ontonotes`.

### Extracting Data
To extract all OntoNotes tasks, run:
```
python extract_ontonotes_all.py --ontonotes /path/to/conll-formatted-ontonotes-5.0 \
  --tasks const coref ner srl \
  --splits train development test conll-2012-test \
  -o $JIANT_DATA_DIR/edges/ontonotes
```
This will write a number of JSON files, one for each split for each task, with names `{task}/{split}.json`.

### Splitting Constituent Data

The constituent data from the script above includes both preterminal (POS tag) and nonterminal (constituent) examples. We can split these into the `edges-nonterminal-ontonotes` and `edges-pos-ontonotes` tasks by running:
```
python jiant/probing/split_constituent_data.py $JIANT_DATA_DIR/edges/ontonotes/const/*.json
```
This will create `*.pos.json` and `*.nonterminal.json` versions of each input file.

## Semantic Proto Roles (SPR)

### SPR1
Tasks: `edges-spr1`

The version of SPR1 distributed on [decomp.io](http://decomp.io/) is difficult to work with directly, because it requires joining with both the Penn Treebank and the PropBank SRL annotations. If you have access to the Penn Treebank ([LDC99T42](https://catalog.ldc.upenn.edu/ldc99t42)), contact Rachel Rudinger or Ian Tenney for a processed copy of the data.

From Rachel's JSON format, you can use a script in this directory to convert to edge probing format:

```
./convert-spr1-rudinger.py -i /path/to/spr1/*.json \
    -o $JIANT_DATA_DIR/edges/spr1
```

You should get files named `spr1.{split}.json` where `split = {train, dev, test}`.

### SPR2
Tasks: `edges-spr2`

Run:
```
pip install conllu
./get_spr2_data.sh $JIANT_DATA_DIR/edges/spr2
```

This downloads both the UD treebank and the annotations and performs a join. See the `get_spr2_data.sh` script for more info. The `conllu` package is required to process the Universal Dependencies source data.


## Definite Pronoun Resolution (DPR)

Tasks: `edges-dpr`

Run:
```
./get_dpr_data.sh $JIANT_DATA_DIR/edges/dpr
```

## Universal Dependencies

Tasks: `edges-dep-labeling-ewt`

Run:
```
./get_ud_data.sh $JIANT_DATA_DIR/edges/dep_ewt
```

This downloads the UD treebank and converts the conllu format to the edge probing format.
