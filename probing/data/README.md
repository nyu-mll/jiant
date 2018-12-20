# Edge Probing Datasets

This directory contains scripts to process the source datasets and generate the JSON data for edge probing tasks. Let `$JIANT_DATA_DIR` be the jiant data directory from the [main README](../../README.md), and make a subdirectory for the edge probing data:

```
mkdir $JIANT_DATA_DIR/edges
```

The resulting JSON has one example per line, with the structure as described in [jiant/probing/README.md](../README.md).

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
This will create `pos/*.json` and `nonterminal/*.json` versions of each input file.

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
