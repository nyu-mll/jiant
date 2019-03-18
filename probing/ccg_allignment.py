import retokenize_edge_data as retokenize
import sys
import argparse
import pandas as pd
import pickle

WORD_TOKENIZATION_INDEX = 2

def get_bpe_tags(text, current_tags, tokenizer_name, tag_dict):
    aligner_fn = retokenize.get_aligner_fn(tokenizer_name)
    assert len(text) == len(current_tags)
    res_tags = []
    introduced_tokenizer_tag = len(tag_dict) - 1
    for i  in range(len(text)):
        token = text[i]
        _, new_toks = aligner_fn(token)
        res_tags.append(tag_dict[current_tags[i]])
        if len(new_toks) > 1:
            for tok in new_toks[1:]:
                res_tags.append(introduced_tokenizer_tag) 
                # based on BERT-paper for wordpiece,, we only predict tag 
                # for the first part of the word.
    _, aligned_text = aligner_fn(" ".join(text))
    assert len(aligned_text) == len(res_tags)
    str_tags = [str(s) for s in res_tags]
    return " ".join(str_tags)

def align_tags_BERT(split, tokenizer_name, data_dir):
    """
        This processes the dataset into the form that edge probing can read in
        and aligns the spain indices from character to tokenizerspan-aligned indices. 
        For example, 
         "I like Bob Sutter yeah " becomes soemthing like
        ["I", "like", "Bob", "Sut", "ter", "yeah"]
        The gold labels given in GAP ha
        s noun index as [7:16], however, with tokenization, we
        want tokenization of [2:4]
        Output: 
        A TSV file readable

        Here, we asign index 0 to be the BPE_tokenized word boundary
    """
    tags_to_id = pickle.load(open("tags_to_id", "rb"))
    ccg_text = pd.read_csv(data_dir+"ccg." + split, names=[ "text", "tags"], delimiter="\t")
    new_pandas = []
    for i in range(len(ccg_text)):
        row = ccg_text.iloc[i]
        text = row['text'].split()
        current_tags = row["tags"].split()
        tags = get_bpe_tags(text, current_tags, tokenizer_name, tags_to_id)
        # mapping between MNE and other. 
        new_pandas.append([row["text"],tags])
    result = pd.DataFrame(new_pandas, columns=["text", "tags"])
    result.to_csv(data_dir+"ccg."+split+"."+tokenizer_name, sep="\t")

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_dir',
        help='directory to save data to',
        type=str,
        default='../data')
    parser.add_argument(
        '-t',
        '--tokenizer',
        help='intended tokenization',
        type=str,
        default='MosesTokenizer')
    args = parser.parse_args(arguments)
    align_tags_BERT("train", args.tokenizer, args.data_dir)
    align_tags_BERT("dev", args.tokenizer, args.data_dir)
    align_tags_BERT("test", args.tokenizer, args.data_dir)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
