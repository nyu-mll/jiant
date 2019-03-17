
"""
Loads gap coreference data and preproceses it, by 
aligning  spans from output of scripts/gap_related_scripts.py

"""
import retokenize_edge_data as retokenize
import sys
import argparse
import pandas as pd
import pickle
from enum import Enum


UNKNOWN = "Unknown"
MASCULINE = "Male"
FEMININE = "Female"


# Mapping of (lowercased) pronoun form to gender value. Note that reflexives
# are not included in GAP, so do not appear here.
PRONOUNS = {
    'she': FEMININE,
    'her': FEMININE,
    'hers': FEMININE,
    'he': MASCULINE,
    'his': MASCULINE,
    'him': MASCULINE,
}

def getEnd(index, word):
    return index  + len(word)

def build_spans_aligned_with_tokenization(text, tokenizer_name, orig_span1_start, orig_span1_end, orig_span2_start, orig_span2_end):
    """
    Builds the alignment while also tokenizing the input piece by piece. 
    """
    if orig_span1_end > orig_span2_start:
        # switch them since the pronoun comes after 
        span2_start = orig_span1_start
        span2_end = orig_span1_end
        span1_start = orig_span2_start
        span1_end = orig_span2_end
    else:
        span1_start = orig_span1_start
        span1_end = orig_span1_end
        span2_start = orig_span2_start
        span2_end = orig_span2_end

    current_tokenization = []
    aligner_fn = retokenize.get_aligner_fn(tokenizer_name)
    text1 = text[:span1_start]
    ta, new_tokens = aligner_fn(text1)
    current_tokenization.extend(new_tokens)
    new_span1start = len(current_tokenization)

    span1 = text[span1_start:span1_end] 
    ta, span_tokens = aligner_fn(span1)
    current_tokenization.extend(span_tokens)
    new_span1end = len(current_tokenization) 
    text2 = text[span1_end:span2_start]
    ta, new_tokens = aligner_fn(text2)
    current_tokenization.extend(new_tokens)
    new_span2start = len(current_tokenization)

    span2 = text[span2_start:span2_end]
    ta, span_tokens = aligner_fn(span2)
    current_tokenization.extend(span_tokens)
    new_span2end = len(current_tokenization)

    text3 = text[span2_end:]
    ta, span_tokens = aligner_fn(text3)
    current_tokenization.extend(span_tokens)

    text = " ".join(current_tokenization)
    if orig_span1_end > orig_span2_start:
        return new_span2start, new_span2end, new_span1start, new_span1end,text
    return  new_span1start, new_span1end, new_span2start, new_span2end, text

def align_spans(split, tokenizer_name, data_dir):
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
    """
    gap_text = pd.read_csv(data_dir+"gap-" + split + ".tsv",  header = 0, delimiter="\t")
    new_pandas = []
    for i in range(len(gap_text)):
        row = gap_text.iloc[i]
        text = row['Text']
        pronoun = row['Pronoun']
        gender = PRONOUNS[pronoun.lower()]
        orig_pronoun_index = row["Pronoun-offset"]
        orig_end_index_prnn = getEnd(orig_pronoun_index, pronoun)
        orig_first_index = row["A-offset"]
        first_word = row["A"]
        orig_end_index = getEnd(orig_first_index, first_word)
        pronoun_index, end_index_prnn, first_index, end_index,text = build_spans_aligned_with_tokenization(text, tokenizer_name, orig_pronoun_index, orig_end_index_prnn, orig_first_index, orig_end_index)
        
        label = str(row["A-coref"]).lower()
        new_pandas.append([text, gender, pronoun_index, end_index_prnn, first_index, end_index, label])

        second_index = row["B-offset"]
        second_word = row["A"]
        end_index_b = getEnd(second_index, second_word)
        _, _, second_index, end_index_b, text = build_spans_aligned_with_tokenization(text, tokenizer_name, orig_pronoun_index, orig_end_index_prnn, second_index, end_index_b)
        label_b = str(row['B-coref']).lower()

        new_pandas.append([text, gender, pronoun_index, end_index_prnn, second_index, end_index_b, label_b])

    result = pd.DataFrame(new_pandas, columns=["text", "gender", "prompt_start_index", "prompt_end_index", "candidate_start_index", "candidate_end_index", "label"])
    result.to_csv(data_dir+"processed/gap-coreference/__"+split+"__"+tokenizer_name, sep="\t")


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
    align_spans("test", args.tokenizer, args.data_dir)
    align_spans("validation", args.tokenizer, args.data_dir)
    align_spans("development", args.tokenizer, args.data_dir)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
