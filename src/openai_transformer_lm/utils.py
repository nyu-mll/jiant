import os

from typing import List

import numpy as np

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
FILL_ID = n_vocab

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

def prep_ids(ids_lists: List[List[int]], n_ctx=512, n_special=3) -> np.ndarray:
    """Prepare IDs for OpenAI Transformer model.
    
    Pads using FILL_ID as start, end, and padding fill,
    and adds positional indices at each position.

    Args:
        ids_list: list of list of integer token ids
        n_ctx: max sequence length
        n_special: number of special tokens to add

    Returns:
        [batch_size, n_ctx, 2] np.ndarray of int32
    """
    x_in = np.zeros((len(ids_lists), n_ctx, 2), dtype=np.int32)
    x_in[:,:,0] = FILL_ID
    x_in[:,:,1] = n_vocab + n_special + np.arange(n_ctx)
    for i, ids in enumerate(ids_lists):
        x_in[i,1:len(ids)+1,0] = ids[:n_ctx-2]
    return x_in

