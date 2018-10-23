# Edge Probing Datasets

This directory contains scripts to process the source datasets and generate the JSON data for edge probing tasks.

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

## OntoNotes
Tasks:
- Constituents / POS: `constituent-ontonotes`, `nonterminal-ontonotes`,
  `pos-ontonotes`
- Entities: `ner-ontonotes`
- SRL: `srl-ontonotes`
- Coreference: `coref-ontonotes-conll`

### Getting OntoNotes Data
Follow the instructions at http://cemantix.org/data/ontonotes.html; you should end up with a folder named `conll-formatted-ontonotes-5.0/`.

If you're working on the JSALT cloud project, you can also download this directly from `gs://jsalt-data/ontonotes`.

### Extracting Data
To extract all datasets, run:
```
python extract_ontonotes_all.py --ontonotes /path/to/conll-formatted-ontonotes-5.0 \
  --tasks const coref ner srl \
  --splits train development test conll-2012-test \
  -o $OUTPUT_DIR
```
This will write a number of JSON files to `$OUTPUT_DIR`, one for each split for each task, with names `$OUTPUT_DIR/{task}.{split}.json`.


## Semantic Proto Roles (SPR)

Tasks: `spr1`, `spr2`

### SPR1

The version of SPR1 distributed on [decomp.io](http://decomp.io/) is difficult to work with directly, because it requires joining with both the Penn Treebank and the PropBank SRL annotations. If you have access to the Penn Treebank ([LDC99T42](https://catalog.ldc.upenn.edu/ldc99t42)), contact Rachel Rudinger or Ian Tenney for a processed copy of the data.

From Rachel's JSON format, you can use a script in this directory to convert to edge probing format:

```
./convert-spr1-rudinger.py -i /path/to/spr1/*.json \
    -o /path/to/probing/data/spr1/
```

You should get files named `spr1.{split}.json` where `split = {train, dev, test}`.

### SPR2

Run:
```
pip install conllu
./get_spr2_data.sh $JIANT_DATA_DIR/spr2
```

This downloads both the UD treebank and the annotations and performs a join. See the `get_spr2_data.sh` script for more info. The `conllu` package is required to process the Universal Dependencies source data.


## Definite Pronoun Resolution (DPR)

Tasks: `dpr`

To get the original data, run `bash get_dpr_data.sh`.
To convert the data, run `python convert-dpr.py`


## Universal Dependencies (TODO: Tom)

Lorem ipsum...

