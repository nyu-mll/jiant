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
python extrace_ontonotes_all.py --ontonotes /path/to/conll-formatted-ontonotes-5.0 \
  --tasks const coref ner srl \
  --splits train development test conll-2012-test \
  -o $OUTPUT_DIR
```
This will write a number of JSON files to `$OUTPUT_DIR`, one for each split for each task, with names `$OUTPUT_DIR/{task}.{split}.json`.


## Semantic Proto Roles v2 (Adam)

Tasks: `edges-spr2`

Run:
```
pip install conllu
./get_spr2_data.sh $JIANT_DATA_DIR/spr2
```

The `conllu` package is required to process the universal dependencies source data.

## Definite Pronoun Resolution (Adam)

Tasks: ` `

To get the original data, run `bash get_dpr_data.sh`.
To convert the data, run `python convert-dpr.py`

## Penn Treebank Constituent Labeling

### Getting the Full PTB Dataset and Generating Edge Probing Data

Follow [this](https://www.ldc.upenn.edu/language-resources/data/obtaining) to obtain the full PTB dataset. After downloading/unzipping `ptb` directory to `/path/to/data`, run `python ptb_process -env /path/to/data` to generate `ptb_train.json`, `ptb_dev.json`, `ptb_dev.full.json`, and `ptb_test.json` (there are python libraries that may need to be downloaded).  Files in the current directory with the same name as the three files generated will be overwritten.

### Split of Penn Treebank (PTB) Dataset

For choice of train/dev/test, we followed [Klein *et al*](http://ilpubs.stanford.edu:8091/~klein/unlexicalized-parsing.pdf).  Note that there is a discrepency in the size of our dev set (first 20 files in section 22 of WSJ section of the Penn treebank), which has 427 sentences compared to 393 as described in Klein *et al*.

## Universal Dependencies (TODO: Tom)

Lorem ipsum...

