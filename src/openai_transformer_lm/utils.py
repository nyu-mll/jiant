import os

from typing import List

from .tf_original import utils as openai_utils
from .tf_original import text_utils as openai_text_utils

openai_data_dir = os.path.join(os.path.dirname(openai_utils.__file__), 
                               "model")
encoder_path = os.path.join(openai_data_dir, "encoder_bpe_40000.json")
bpe_path = os.path.join(openai_data_dir, "vocab_40000.bpe")

text_encoder = openai_text_utils.TextEncoder(encoder_path, bpe_path)
encoder_dict = text_encoder.encoder  # just a dict of text -> id
reverse_encoder_dict = {v:k for k,v in encoder_dict.items()}
n_vocab = len(encoder_dict)
assert n_vocab == 40478

def encode(sentences: List[str]) -> List[List[int]]:
    return text_encoder.encode(sentences)

def lookup_ids(ids: List[int]) -> List[str]:
    return [reverse_encoder_dict[i] for i in ids]

def undo_wpm(pieces: List[str]) -> str:
    return "".join(pieces).replace("</w>", " ").strip(" ")

def decode_partial(ids: List[List[int]]) -> List[List[str]]:
    """Decode ids, but not undo WPM."""
    return map(lookup_ids, ids)

def decode_full(ids: List[List[int]]) -> List[str]:
    """Decode ids to strings."""
    return map(undo_wpm, map(lookup_ids, ids))

def tokenize(sentence: str):
    ids = encode([sentence])[0]
    return lookup_ids(ids)
