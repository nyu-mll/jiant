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
from jiant.pytorch_transformers_interface import input_module_uses_pytorch_transformers
from pytorch_transformers import (
    BertTokenizer,
    RobertaTokenizer,
    XLNetTokenizer,
    OpenAIGPTTokenizer,
    GPT2Tokenizer,
    TransfoXLTokenizer,
    XLMTokenizer,
)


class Tokenizer(object):
    def tokenize(self, sentence):
        raise NotImplementedError


def select_tokenizer(args):
    """
        Select a sane default tokenizer.
    """
    if args.tokenizer == "auto":
        if input_module_uses_pytorch_transformers(args.input_module):
            tokenizer_name = args.input_module
        else:
            tokenizer_name = "MosesTokenizer"
    else:
        tokenizer_name = args.tokenizer
    return tokenizer_name


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


@functools.lru_cache(maxsize=8, typed=False)
def get_tokenizer(tokenizer_name):
    log.info(f"\tLoading Tokenizer {tokenizer_name}")
    if tokenizer_name.startswith("bert-"):
        do_lower_case = tokenizer_name.endswith("uncased")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=do_lower_case)
    elif tokenizer_name.startswith("roberta-"):
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
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
    elif tokenizer_name.startswith("xlm-"):
        tokenizer = XLMTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name == "MosesTokenizer":
        tokenizer = MosesTokenizer()
    elif tokenizer_name == "":
        tokenizer = SpaceTokenizer()
    else:
        tokenizer = None
    return tokenizer
