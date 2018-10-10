import os
import random
import logging as log

from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn

from .tf_original import utils as openai_utils
from .tf_original.text_utils import TextEncoder

from .pytorch_huggingface import model_pytorch

openai_data_dir = os.path.join(os.path.dirname(openai_utils.__file__),
                               "model")
encoder_path = os.path.join(openai_data_dir, "encoder_bpe_40000.json")
bpe_path = os.path.join(openai_data_dir, "vocab_40000.bpe")

text_encoder = TextEncoder(encoder_path, bpe_path)
encoder_dict = text_encoder.encoder  # just a dict of text -> id
reverse_encoder_dict = {v:k for k,v in encoder_dict.items()}
N_VOCAB = len(encoder_dict)
assert N_VOCAB == 40478
FILL_ID = N_VOCAB

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
    x_in[:,:,1] = N_VOCAB + n_special + np.arange(n_ctx)
    for i, ids in enumerate(ids_lists):
        x_in[i,1:len(ids)+1,0] = ids[:n_ctx-2]
    return x_in


class OpenAIEmbedderModule(nn.Module):
    def __init__(self, args, n_special=3, n_ctx=512):
        super(OpenAIEmbedderModule, self).__init__()
        self.model_cfg = model_pytorch.DEFAULT_CONFIG
        self.n_special = n_special  # number of special tokens
        self.n_ctx = n_ctx  # max context width (seq len)

        full_emb_vocab = N_VOCAB + self.n_special + self.n_ctx
        self.model = model_pytorch.TransformerModel(self.model_cfg,
                                                    vocab=full_emb_vocab)

        # Need specific seed to reproduce results.
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        loader_args = dict(n_special=n_special)
        # Path to model weights
        loader_args['path'] = openai_data_dir + "/"
        # Path to variable name mapping
        loader_args['path_names'] = os.path.dirname(model_pytorch.__file__) + "/"
        # Load pretrained weights from disk
        log.info("Loading OpenAI transformer model from %s", loader_args['path'])
        model_pytorch.load_openai_pretrained_model(self.model, **loader_args)
        log.info("Loaded OpenAI transformer model.")

        # Set trainability of this module.
        for param in self.model.parameters():
            param.requires_grad = bool(args.openai_transformer_fine_tune)

    def forward(self, sent: Dict[str, torch.LongTensor],
                unused_task_name: str="") -> torch.FloatTensor:
        """ Run transformer to get hidden states.

        Args:
            sent: batch dictionary

        Returns:
            h: [batch_size, n_ctx, d_emb]
        """
        assert "openai_bpe_pretokenized" in sent
        var_ids = sent["openai_bpe_pretokenized"]

        # Model has fixed, learned positional component :(, so we must pass a
        # block of exactly n_ctx length.
        ids = torch.zeros(var_ids.size()[0], self.n_ctx, dtype=var_ids.dtype,
                          device=var_ids.device)
        fill_len = min(var_ids.size()[1], self.n_ctx)
        ids[:,:fill_len] = var_ids[:,:fill_len]
        # "Correct" ids to account for different indexing between OpenAI and
        # AllenNLP.
        # The AllenNLP indexer adds a '@@UNKNOWN@@' token to the
        # beginning of the vocabulary, *and* treats that as index 1 (index 0 is
        # reserved for padding). In the OpenAI model, index 0 is their own
        # <unk> token and index 1 is the first real wordpiece, ".".
        # So, to correct things we should subtract 2, so that AllenNLP's
        # handling of "." (index 3) gets mapped to what the OpenAI model
        # expects (index 1).
        ids[ids == 0] = FILL_ID + 2
        ids -= 2

        # Generate positional indices.
        pos_ids = torch.arange(self.n_ctx, dtype=torch.int64,
                               device=ids.device).repeat(ids.size()[0], 1)
        pos_ids = N_VOCAB + self.n_special + pos_ids

        # x is [batch_size, n_ctx, 2]
        x = torch.stack([ids, pos_ids], dim=2)
        # h is [batch_size, n_ctx, d_emb]
        h = self.model(x)

        # Truncate back to the original ids length, for compatiblity with the
        # rest of our embedding models. This only drops padding
        # representations.
        h_trunc = h[:,:var_ids.size()[1],:]
        return h_trunc

    def get_output_dim(self):
        return self.model_cfg['n_embd']


