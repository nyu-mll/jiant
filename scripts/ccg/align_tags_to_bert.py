import argparse
import json
import os
import sys

import pandas as pd

from jiant.utils.retokenize import get_aligner_fn


"""
Usage:
    Run the below command from the root directory
    python -m scripts.ccg.align_tags_to_bert --data_dir {path/to/ccg/files} -t {tokenizer_name}

The input file should be in csv form, with text and tags columns.

The output format has columns text and tag, which is a string of space delimited numbers.
This preprocessing file will preprocess the CCG data using the tokenizer,
saving it alongside the original files.

This file introduces a new tag to sub-words (if the tokenizer splits a word.
Currently, this supports BERT tokenization.)
For example,
[Mr., Porter] -> [Mr, ., Por, ter]. Thus, if Mr. was given a tag of 5 and Porter 6
in the original CCG, then the alligned tags will be [5, 1363, 6, 1363], where 1363 indicates
a subpiece of a word that has been split due to tokenization.
"""


def get_tags(text, current_tags, tokenizer_name, tag_dict):
    aligner_fn = get_aligner_fn(tokenizer_name)
    assert len(text) == len(current_tags)
    introduced_tokenizer_tag = len(tag_dict)
    token_aligner, aligned_text = aligner_fn(" ".join(text))
    aligned_tags = [introduced_tokenizer_tag for token in aligned_text]
    for text_idx, text_tag in enumerate(current_tags):
        aligned_idx = token_aligner.project_tokens(text_idx)[0]
        aligned_tags[aligned_idx] = tag_dict[text_tag]
    str_tags = [str(s) for s in aligned_tags]
    return " ".join(str_tags)


def align_tags_BERT(dataset, tokenizer_name, tags_to_id):
    new_pandas = []
    for i in range(len(dataset)):
        row = dataset.iloc[i]
        text = row["text"].split()
        current_tags = row["tags"].split()
        tags = get_tags(text, current_tags, tokenizer_name, tags_to_id)
        new_pandas.append([row["text"], tags])
    result = pd.DataFrame(new_pandas, columns=["text", "tags"])
    return result


def align_ccg(split, tokenizer_name, data_dir):
    """
        We align BERT tags such that any introduced tokens introudced
        by BERT will be assigned a special tag, which later on in
        preprocessing/task building will be converted into a tag mask (so that
        we do not take into account the model prediction for tokens introduced by
        the tokenizer).

        Args
        --------------
        split: str,
        tokenizer_name: str,
        data_dir: str,

        Returns
        --------------
        None, saves tag alligned files to same directory as the original file.
    """
    tags_to_id = json.load(open(os.path.join(data_dir, "tags_to_id.json"), "r"))
    ccg_text = pd.read_csv(
        os.path.join(data_dir, "ccg." + split), names=["text", "tags"], delimiter="\t"
    )
    result = align_tags_BERT(ccg_text, tokenizer_name, tags_to_id)
    result.to_csv(os.path.join(data_dir, "ccg." + split + "." + tokenizer_name), sep="\t")


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", help="directory to save data to", type=str, default="../data"
    )
    parser.add_argument(
        "-t", "--tokenizer", help="intended tokenization", type=str, default="MosesTokenizer"
    )
    args = parser.parse_args(arguments)
    align_ccg("train", args.tokenizer, args.data_dir)
    align_ccg("dev", args.tokenizer, args.data_dir)
    align_ccg("test", args.tokenizer, args.data_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
