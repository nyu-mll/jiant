"""
Functions having to do with loading data from output of
files downloaded in scripts/download_data_glue.py

"""
import codecs
import csv
import json
import numpy as np
import pandas as pd
from allennlp.data import vocabulary

from jiant.utils.tokenizers import get_tokenizer
from jiant.utils.retokenize import realign_spans


def load_span_data(tokenizer_name, file_name, label_fn=None, has_labels=True):
    """
    Load a span-related task file in .jsonl format, does re-alignment of spans, and tokenizes
    the text.
    Re-alignment of spans involves transforming the spans so that it matches the text after
    tokenization.
    For example, given the original text: [Mr., Porter, is, nice] and bert-base-cased
    tokenization, we get [Mr, ., Por, ter, is, nice ]. If the original span indices was [0,2],
    under the new tokenization, it becomes [0, 3].
    The task file should of be of the following form:
        text: str,
        label: bool
        target: dict that contains the spans
    Args:
        tokenizer_name: str,
        file_name: str,
        label_fn: function that expects a row and outputs a transformed row with labels
          transformed.
    Returns:
        List of dictionaries of the aligned spans and tokenized text.
    """
    rows = pd.read_json(file_name, lines=True)
    # realign spans
    rows = rows.apply(lambda x: realign_spans(x, tokenizer_name), axis=1)
    if has_labels is False:
        rows["label"] = 0
    elif label_fn is not None:
        rows["label"] = rows["label"].apply(label_fn)
    return list(rows.T.to_dict().values())


def load_pair_nli_jsonl(data_file, tokenizer_name, max_seq_len, targ_map):
    """
    Loads a pair NLI task.

    Parameters
    -----------------
    data_file: path to data file,
    tokenizer_name: str,
    max_seq_len: int,
    targ_map: a dictionary that maps labels to ints

    Returns
    -----------------
    sent1s: list of strings of tokenized first sentences,
    sent2s: list of strings of tokenized second sentences,
    trgs: list of ints of labels,
    idxs: list of ints
    """
    data = [json.loads(d) for d in open(data_file, encoding="utf-8")]
    sent1s, sent2s, trgs, idxs, pair_ids = [], [], [], [], []
    for example in data:
        sent1s.append(tokenize_and_truncate(tokenizer_name, example["premise"], max_seq_len))
        sent2s.append(tokenize_and_truncate(tokenizer_name, example["hypothesis"], max_seq_len))
        trg = targ_map[example["label"]] if "label" in example else 0
        trgs.append(trg)
        idxs.append(example["idx"])
        if "pair_id" in example:
            pair_ids.append(example["pair_id"])
    return [sent1s, sent2s, trgs, idxs, pair_ids]


def load_tsv(
    tokenizer_name,
    data_file,
    max_seq_len,
    label_idx=2,
    s1_idx=0,
    s2_idx=1,
    label_fn=None,
    skip_rows=0,
    return_indices=False,
    delimiter="\t",
    quote_level=csv.QUOTE_NONE,
    filter_idx=None,
    has_labels=True,
    filter_value=None,
    tag_vocab=None,
    tag2idx_dict=None,
):
    """
    Load a tsv.

    To load only rows that have a certain value for a certain columnn, set filter_idx and
    filter_value (for example, for mnli-fiction we want rows where the genre column has
    value 'fiction').

    Args:
        tokenizer_name (str): The name of the tokenizer to use (see defaluts.conf for values).
        data_file (str): The path to the file to read.
        max_seq_len (int): The maximum number of tokens to keep after tokenization, per text field.
            Start and end symbols are introduced before tokenization, and are counted, so we will
            keep max_seq_len - 2 tokens *of text*.
        label_idx (int|None): The column index for the label field, if present.
        s1_idx (int): The column index for the first text field.
        s2_idx (int|None): The column index for the second text field, if present.
        label_fn (fn: str -> int|None): A function to map items in column label_idx to int-valued
            labels.
        skip_rows (int|list): Skip this many header rows or skip these specific row indices.
        has_labels (bool): If False, don't look for labels at position label_idx.
        filter_value (str|None): The value in which we want filter_idx to be equal to.
        filter_idx (int|None): The column index in which to look for filter_value.
        tag_vocab (allennlp vocabulary): In some datasets, examples are attached to tags, and we
            need to know the results on examples with certain tags, this is a vocabulary for
            tracking tags in a dataset across splits
        tag2idx_dict (dict<string, int>): The tags form a two-level hierarchy, each fine tag belong
            to a coarse tag. In the tsv, each coarse tag has one column, the content in that column
            indicates what fine tags(seperated by ;) beneath that coarse tag the examples have. 
            tag2idx_dict is a dictionary to map coarse tag to the index of corresponding column.
            e.g. if we have two coarse tags: source at column 0, topic at column 1; and four fine
            tags: wiki, reddit beneath source, and economics, politics beneath topic. The tsv will
            be: | wiki  | economics;politics|, with the tag2idx_dict as {"source": 0, "topic": 1}
                | reddit| politics          |

    Returns:
        List of first and second sentences, labels, and if applicable indices
    """

    # TODO(Yada): Instead of index integers, adjust this to pass in column names
    # get the first row as the columns to pass into the pandas reader
    # This reads the data file given the delimiter, skipping over any rows
    # (usually header row)
    rows = pd.read_csv(
        data_file,
        sep=delimiter,
        error_bad_lines=False,
        header=None,
        skiprows=skip_rows,
        quoting=quote_level,
        keep_default_na=False,
        encoding="utf-8",
    )

    if filter_idx and filter_value:
        rows = rows[rows[filter_idx] == filter_value]
    # Filter for sentence1s that are of length 0
    # Filter if row[targ_idx] is nan
    mask = rows[s1_idx].str.len() > 0
    if s2_idx is not None:
        mask = mask & (rows[s2_idx].str.len() > 0)
    if has_labels:
        mask = mask & rows[label_idx].notnull()
    rows = rows.loc[mask]
    sent1s = rows[s1_idx].apply(lambda x: tokenize_and_truncate(tokenizer_name, x, max_seq_len))
    if s2_idx is None:
        sent2s = pd.Series()
    else:
        sent2s = rows[s2_idx].apply(lambda x: tokenize_and_truncate(tokenizer_name, x, max_seq_len))

    label_fn = label_fn if label_fn is not None else (lambda x: x)
    if has_labels:
        labels = rows[label_idx].apply(lambda x: label_fn(x))
    else:
        # If dataset doesn't have labels, for example for test set, then mock labels
        labels = np.zeros(len(rows), dtype=int)
    if tag2idx_dict is not None:
        # -2 offset to cancel @@unknown@@ and @@padding@@ in vocab
        def tags_to_tids(coarse_tag, fine_tags):
            return (
                []
                if pd.isna(fine_tags)
                else (
                    [tag_vocab.add_token_to_namespace(coarse_tag) - 2]
                    + [
                        tag_vocab.add_token_to_namespace("%s__%s" % (coarse_tag, fine_tag)) - 2
                        for fine_tag in fine_tags.split(";")
                    ]
                )
            )

        tid_temp = [
            rows[idx].apply(lambda x: tags_to_tids(coarse_tag, x)).tolist()
            for coarse_tag, idx in tag2idx_dict.items()
        ]
        tagids = [[tid for column in tid_temp for tid in column[idx]] for idx in range(len(rows))]
    if return_indices:
        idxs = rows.index.tolist()
        # Get indices of the remaining rows after filtering
        return sent1s.tolist(), sent2s.tolist(), labels.tolist(), idxs
    elif tag2idx_dict is not None:
        return sent1s.tolist(), sent2s.tolist(), labels.tolist(), tagids
    else:
        return sent1s.tolist(), sent2s.tolist(), labels.tolist()


def load_diagnostic_tsv(
    tokenizer_name,
    data_file,
    max_seq_len,
    label_col,
    s1_col="",
    s2_col="",
    label_fn=None,
    skip_rows=0,
    delimiter="\t",
):
    """Load a tsv and indexes the columns from the diagnostic tsv.
        This is only used for GLUEDiagnosticTask right now.
    Args:
        data_file: string
        max_seq_len: int
        s1_col: string
        s2_col: string
        label_col: string
        label_fn: function
        skip_rows: list of ints
        delimiter: string
    Returns:
        A dictionary of the necessary indexed fields, the tokenized sent1 and sent2
        and indices
        Note: If a field in a particular row in the dataset is empty, we return []
        for that field for that row, otherwise we return an array of ints (indices)
        Else, we return an array of indices
    """
    # TODO: Abstract indexing layer from this function so that MNLI-diagnostic
    # calls load_tsv
    assert (
        len(s1_col) > 0 and len(label_col) > 0
    ), "Make sure you passed in column names for sentence 1 and labels"
    rows = pd.read_csv(
        data_file, sep=delimiter, error_bad_lines=False, quoting=csv.QUOTE_NONE, encoding="utf-8"
    )
    rows = rows.fillna("")

    def targs_to_idx(col_name):
        # This function builds the index to vocab (and its inverse) mapping
        values = set(rows[col_name].values)
        vocab = vocabulary.Vocabulary(counter=None, non_padded_namespaces=[col_name])
        for value in values:
            vocab.add_token_to_namespace(value, col_name)
        idx_to_word = vocab.get_index_to_token_vocabulary(col_name)
        word_to_idx = vocab.get_token_to_index_vocabulary(col_name)
        rows[col_name] = rows[col_name].apply(lambda x: [word_to_idx[x]] if x != "" else [])
        return word_to_idx, idx_to_word, rows[col_name]

    sent1s = rows[s1_col].apply(lambda x: tokenize_and_truncate(tokenizer_name, x, max_seq_len))
    sent2s = rows[s2_col].apply(lambda x: tokenize_and_truncate(tokenizer_name, x, max_seq_len))
    labels = rows[label_col].apply(lambda x: label_fn(x))
    # Build indices for field attributes
    lex_sem_to_ix_dic, ix_to_lex_sem_dic, lex_sem = targs_to_idx("Lexical Semantics")
    pr_ar_str_to_ix_di, ix_to_pr_ar_str_dic, pr_ar_str = targs_to_idx(
        "Predicate-Argument Structure"
    )
    logic_to_ix_dic, ix_to_logic_dic, logic = targs_to_idx("Logic")
    knowledge_to_ix_dic, ix_to_knowledge_dic, knowledge = targs_to_idx("Knowledge")
    idxs = rows.index

    return {
        "sents1": sent1s.tolist(),
        "sents2": sent2s.tolist(),
        "targs": labels.tolist(),
        "idxs": idxs.tolist(),
        "lex_sem": lex_sem.tolist(),
        "pr_ar_str": pr_ar_str.tolist(),
        "logic": logic.tolist(),
        "knowledge": knowledge.tolist(),
        "ix_to_lex_sem_dic": ix_to_lex_sem_dic,
        "ix_to_pr_ar_str_dic": ix_to_pr_ar_str_dic,
        "ix_to_logic_dic": ix_to_logic_dic,
        "ix_to_knowledge_dic": ix_to_knowledge_dic,
    }


def get_tag_list(tag_vocab):
    """
    retrieve tag strings from the tag vocab object
    Args:
        tag_vocab: the vocab that contains all tags
    Returns:
        tag_list: a list of "coarse__fine" tag strings
    """
    # get dictionary from allennlp vocab, neglecting @@unknown@@ and
    # @@padding@@
    tid2tag_dict = {
        key - 2: tag
        for key, tag in tag_vocab.get_index_to_token_vocabulary().items()
        if key - 2 >= 0
    }
    tag_list = [
        tid2tag_dict[tid].replace(":", "_").replace(", ", "_").replace(" ", "_").replace("+", "_")
        for tid in range(len(tid2tag_dict))
    ]
    return tag_list


def tokenize_and_truncate(tokenizer_name, sent, max_seq_len):
    """Truncate and tokenize a sentence or paragraph."""
    max_seq_len -= 2  # For boundary tokens.
    tokenizer = get_tokenizer(tokenizer_name)

    if isinstance(sent, str):
        return tokenizer.tokenize(sent)[:max_seq_len]
    elif isinstance(sent, list):
        assert isinstance(sent[0], str), "Invalid sentence found!"
        return sent[:max_seq_len]
