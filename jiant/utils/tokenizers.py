"""
Tokenizer class
To add a tokenizer, add to the below inherting from
main Tokenizer class.
"""
import functools
import logging as log
import os

from sacremoses import MosesDetokenizer
from sacremoses import MosesTokenizer as SacreMosesTokenizer
from nltk.tokenize.simple import SpaceTokenizer
from jiant.huggingface_transformers_interface import input_module_uses_transformers
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    AlbertTokenizer,
    XLNetTokenizer,
    OpenAIGPTTokenizer,
    GPT2Tokenizer,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMRobertaTokenizer,
)


class Tokenizer(object):
    def tokenize(self, sentence):
        raise NotImplementedError


def select_tokenizer(args):
    """
        Select a sane default tokenizer.
    """
    if args.tokenizer == "auto":
        if input_module_uses_transformers(args.input_module):
            tokenizer_name = args.input_module
        else:
            tokenizer_name = "MosesTokenizer"
    else:
        tokenizer_name = args.tokenizer
    return tokenizer_name


class SplitCharsTokenizer(Tokenizer):
    """
        This tokenizer splits a string (sentence or word) into individual characters.
    """

    def __init__(self):
        super().__init__()

    def tokenize(self, sequence):
        return list(sequence)

    def detokenize(self, tokens):
        return "".join(tokens)


class MosesTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self._tokenizer = SacreMosesTokenizer()
        self._detokenizer = MosesDetokenizer()

    def tokenize(self, sentence):
        return self._tokenizer.tokenize(sentence)

    def detokenize(self, tokens):
        """Unescape Moses punctuation tokens.

        Replaces escape sequences like &#91; with the original characters
        (such as '['), so they better align to the original text.
        """
        return [self._detokenizer.unescape_xml(t) for t in tokens]

    def detokenize_ptb(self, tokens):
        # Not a perfect detokenizer, but a "good-enough" stand in.
        rep_dict = {
            "-LSB-": "[",
            "-RSB-": "]",
            "-LRB-": "(",
            "-RRB-": ")",
            "-LCB-": "{",
            "-RCB-": "}",
            "``": '"',
            "''": '"',
        }
        str1 = self._detokenizer.detokenize(replace_list(tokens, rep_dict))
        return str1


@functools.lru_cache(maxsize=8, typed=False)
def get_tokenizer(tokenizer_name):
    log.info(f"\tLoading Tokenizer {tokenizer_name}")
    if tokenizer_name.startswith("bert-"):
        do_lower_case = tokenizer_name.endswith("uncased")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=do_lower_case)
    elif tokenizer_name.startswith("roberta-"):
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("albert-"):
        tokenizer = AlbertTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("xlnet-"):
        do_lower_case = tokenizer_name.endswith("uncased")
        tokenizer = XLNetTokenizer.from_pretrained(tokenizer_name, do_lower_case=do_lower_case)
    elif tokenizer_name.startswith("openai-gpt"):
        tokenizer = OpenAIGPTTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("transfo-xl-"):
        # TransformerXL is trained on data pretokenized with MosesTokenizer
        tokenizer = MosesTokenizer()
    elif tokenizer_name.startswith("xlm-mlm-") or tokenizer_name.startswith("xlm-clm-"):
        tokenizer = XLMTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("xlm-roberta-"):
        tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name == "MosesTokenizer":
        tokenizer = MosesTokenizer()
    elif tokenizer_name == "SplitChars":
        tokenizer = SplitCharsTokenizer()
    elif tokenizer_name == "":
        tokenizer = SpaceTokenizer()
    else:
        tokenizer = None
    return tokenizer


def bert_get_tokenized_string_span_map(text, b_tokens, verbose=False):
    """
    Given a string, an a BERT tokenization of the string, returns list of
        [
            bert_token,
            start char index of token in string,
            (exclusive) end char index of token in string,
        ]
    There is some fuzziness around assignment of spaces (particularly because of UNK tokens)
      but the spans should be contiguous.
    """
    b_token_char_indices = []
    text_i = 0
    for b_token in b_tokens:
        stripped_b_token = b_token.replace("##", "")
        if b_token == "[UNK]":
            continue

        found_char_i = text[text_i:].find(stripped_b_token)
        b_token_char_indices.append(text_i + found_char_i)
        text_i += len(stripped_b_token) + found_char_i
    b_token_char_indices.append(len(text))

    result = []
    b_token_char_indices_i = -1
    end = 0
    for i, b_token in enumerate(b_tokens):
        prev_end = end

        if b_token == "[UNK]":
            start = prev_end
        else:
            b_token_char_indices_i += 1
            start = b_token_char_indices[b_token_char_indices_i]

        if i == len(b_tokens) - 1:
            end = len(text)
        elif b_token == "[UNK]":
            end = b_token_char_indices[b_token_char_indices_i + 1]
        elif b_token != "[UNK]" and b_tokens[i + 1] != "[UNK]":
            end = b_token_char_indices[b_token_char_indices_i + 1]
        elif b_tokens[i + 1] == "[UNK]":
            end = start + len(b_token)
        else:
            raise RuntimeError()

        if verbose:
            print(b_token, start, end, repr(text[start:end]))
        result.append((b_token, start, end))
    return result


def replace_list(ls, d):
    return [d.get(elem, elem) for elem in ls]
