"""
Tokenizer class
To add a tokenizer, add to the below inherting from
main Tokenizer class.
"""
import functools
import logging as log
import os

from nltk.tokenize.moses import MosesDetokenizer
from nltk.tokenize.moses import MosesTokenizer as NLTKMosesTokenizer
from nltk.tokenize.simple import SpaceTokenizer


class Tokenizer(object):
    def tokenize(self, sentence):
        raise NotImplementedError


class OpenAIBPETokenizer(Tokenizer):
    # TODO: Add detokenize method to OpenAIBPE class
    def __init__(self):
        super().__init__()
        from ..openai_transformer_lm.pytorch_huggingface import utils as openai_utils
        from ..openai_transformer_lm.pytorch_huggingface.text_utils import TextEncoder

        OPENAI_DATA_DIR = os.path.join(os.path.dirname(openai_utils.__file__), "model")
        ENCODER_PATH = os.path.join(OPENAI_DATA_DIR, "encoder_bpe_40000.json")
        BPE_PATH = os.path.join(OPENAI_DATA_DIR, "vocab_40000.bpe")
        self._tokenizer = TextEncoder(ENCODER_PATH, BPE_PATH)
        self._encoder_dict = self._tokenizer.encoder
        self._reverse_encoder_dict = {v: k for k, v in self._encoder_dict.items()}

    def lookup_ids(self, ids):
        return [self._reverse_encoder_dict[i] for i in ids]

    def encode(self, sentences):
        return self._tokenizer.encode(sentences)

    def tokenize(self, sentence):
        ids = self.encode([sentence])[0]
        return self.lookup_ids(ids)


class MosesTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self._tokenizer = NLTKMosesTokenizer()
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
        from pytorch_pretrained_bert import BertTokenizer

        do_lower_case = tokenizer_name.endswith("uncased")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=do_lower_case)
    elif tokenizer_name == "OpenAI.BPE":
        tokenizer = OpenAIBPETokenizer()
    elif tokenizer_name == "MosesTokenizer":
        tokenizer = MosesTokenizer()
    elif tokenizer_name == "":
        tokenizer = SpaceTokenizer()
    else:
        tokenizer = None
    return tokenizer
