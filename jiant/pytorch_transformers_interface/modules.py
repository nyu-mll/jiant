import copy
import logging as log
import os
from typing import Dict

import torch
import torch.nn as nn
from allennlp.modules import scalar_mix

import pytorch_transformers

from jiant.preprocess import parse_task_list_arg
from jiant.utils import utils


class PytorchTransformersEmbedderModule(nn.Module):
    """ Shared code for pytorch_transformers wrappers.

    Subclasses share a good deal of code, but have a number of subtle differences due to different
    APIs from pytorch_transfromers.
    """

    def __init__(self, args):
        super(PytorchTransformersEmbedderModule, self).__init__()

        self.cache_dir = os.getenv(
            "PYTORCH_PRETRAINED_BERT_CACHE",
            os.path.join(args.exp_dir, "pytorch_transformers_cache"),
        )
        utils.maybe_make_dir(self.cache_dir)

        self.embeddings_mode = args.pytorch_transformers_output_mode

        # Integer token indices for special symbols.
        self._sep_id = None
        self._cls_id = None
        self._pad_id = None

        # If set, treat these special tokens as part of input segments other than A/B.
        self._SEG_ID_CLS = None
        self._SEG_ID_SEP = None

    def parameter_setup(self, args):
        # Set trainability of this module.
        for param in self.model.parameters():
            param.requires_grad = bool(args.transfer_paradigm == "finetune")

        self.num_layers = self.model.config.num_hidden_layers
        if args.pytorch_transformers_max_layer >= 0:
            self.max_layer = args.pytorch_transformers_max_layer
            assert self.max_layer <= self.num_layers
        else:
            self.max_layer = self.num_layers

        # Configure scalar mixing, ELMo-style.
        if self.embeddings_mode == "mix":
            if args.transfer_paradigm == "frozen":
                log.warning(
                    "NOTE: pytorch_transformers_output_mode='mix', so scalar "
                    "mixing weights will be fine-tuned even if BERT "
                    "model is frozen."
                )
            # TODO: if doing multiple target tasks, allow for multiple sets of
            # scalars. See the ELMo implementation here:
            # https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L115
            assert len(parse_task_list_arg(args.target_tasks)) <= 1, (
                "pytorch_transformers_output_mode='mix' only supports a single set of "
                "scalars (but if you need this feature, see the TODO in "
                "the code!)"
            )
            # Always have one more mixing weight, for lexical layer.
            self.scalar_mix = scalar_mix.ScalarMix(self.max_layer + 1, do_layer_norm=False)

    def prepare_output(self, lex_seq, hidden_states, mask):
        """
        Convert the output of the pytorch_transformers module to a vector sequence as expected by jiant.

        args:
            lex_seq: The sequence of input word embeddings as a tensor (batch_size, sequence_length, hidden_size).
                     Used only if embeddings_mode = "only".
            hidden_states: A list of sequences of model hidden states as tensors (batch_size, sequence_length, hidden_size).
            mask: A tensor with 1s in positions corresponding to non-padding tokens (batch_size, sequence_length).

        """
        available_layers = hidden_states[: self.max_layer + 1]

        if self.embeddings_mode in ["none", "top"]:
            h = available_layers[-1]
        elif self.embeddings_mode == "only":
            h = lex_seq
        elif self.embeddings_mode == "cat":
            h = torch.cat([available_layers[-1], lex_seq], dim=2)
        elif self.embeddings_mode == "mix":
            h = self.scalar_mix(available_layers, mask=mask)
        else:
            raise NotImplementedError(f"embeddings_mode={self.embeddings_mode}" " not supported.")

        # <float32> [batch_size, var_seq_len, output_dim]
        return h

    def get_output_dim(self):
        if self.embeddings_mode == "cat":
            return 2 * self.model.config.hidden_size
        else:
            return self.model.config.hidden_size

    def get_seg_ids(self, token_ids):
        """ Dynamically build the segment IDs for a concatenated pair of sentences
        Searches for index _sep_id in the tensor. Supports BERT or XLNet-style padding.
        Sets padding tokens to segment zero.

        args:
            token_ids (torch.LongTensor): batch of token IDs

        returns:
            seg_ids (torch.LongTensor): batch of segment IDs

        example:
        > sents = ["[CLS]", "I", "am", "a", "cat", ".", "[SEP]", "You", "like", "cats", "?", "[SEP]", "[PAD]"]
        > token_tensor = torch.Tensor([[vocab[w] for w in sent]]) # a tensor of token indices
        > seg_ids = get_seg_ids(token_tensor)
        > assert seg_ids == torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        """

        sep_idxs = (token_ids == self._sep_id).nonzero()[:, 1]
        seg_ids = torch.zeros_like(token_ids)
        for row_idx, row in enumerate(token_ids):
            sep_idxs = (row == self._sep_id).nonzero()
            seg = 0
            prev_sep_idx = -1
            for sep_idx in sep_idxs:
                seg_ids[row_idx, prev_sep_idx + 1 : sep_idx + 1].fill_(seg)
                seg = 1 - seg  # Alternate.
                prev_sep_idx = sep_idx

        if self._SEG_ID_CLS is not None:
            seg_ids[token_ids == self._cls_id] = self._SEG_ID_CLS

        if self._SEG_ID_SEP is not None:
            seg_ids[token_ids == self._sep_id] = self._SEG_ID_SEP

        return seg_ids


class BertEmbedderModule(PytorchTransformersEmbedderModule):
    """ Wrapper for BERT module to fit into jiant APIs. """

    def __init__(self, args):
        super(BertEmbedderModule, self).__init__(args)

        self.model = pytorch_transformers.BertModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )

        tokenizer = pytorch_transformers.BertTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, do_lower_case="uncased" in args.tokenizer
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.
        self._sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
        self._cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
        self._pad_id = tokenizer.convert_tokens_to_ids("[PAD]")

        self.parameter_setup(args)

    @staticmethod
    def apply_boundary_tokens(s1, s2=None):
        # BERT-style boundary token padding on string token sequences
        if s2:
            return ["[CLS]"] + s1 + ["[SEP]"] + s2 + ["[SEP]"]
        else:
            return ["[CLS]"] + s1 + ["[SEP]"]

    def forward(
        self, sent: Dict[str, torch.LongTensor], unused_task_name: str = ""
    ) -> torch.FloatTensor:
        """ Run BERT to get hidden states.

        This forward method does preprocessing on the go,
        changing token IDs from preprocessed bert to
        what AllenNLP indexes.

        Args:
            sent: batch dictionary

        Returns:
            h: [batch_size, seq_len, d_emb]
        """
        assert "pytorch_transformers_wpm_pretokenized" in sent
        # <int32> [batch_size, var_seq_len]
        ids = sent["pytorch_transformers_wpm_pretokenized"]
        # BERT supports up to 512 tokens; see section 3.2 of https://arxiv.org/pdf/1810.04805.pdf
        assert ids.size()[1] <= 512

        mask = ids != 0
        # "Correct" ids to account for different indexing between BERT and
        # AllenNLP.
        # The AllenNLP indexer adds a '@@UNKNOWN@@' token to the
        # beginning of the vocabulary, *and* treats that as index 1 (index 0 is
        # reserved for padding).
        ids[ids == 0] = self._pad_id + 2  # Shift the indices that were at 0 to become 2.
        # Index 1 should never be used since the BERT WPM uses its own
        # unk token, and handles this at the string level before indexing.
        assert (ids > 1).all()
        ids -= 2  # shift indices to match BERT wordpiece embeddings

        if self.embeddings_mode not in ["none", "top"]:
            # This is redundant with the lookup inside BertModel,
            # but doing so this way avoids the need to modify the BertModel
            # code.
            # Extract lexical embeddings
            lex_seq = self.model.embeddings.word_embeddings(ids)
            lex_seq = self.model.embeddings.LayerNorm(lex_seq)
            hidden_states = []  # dummy; should not be accessed.
            # following our use of the OpenAI model, don't use dropout for
            # probing. If you would like to use dropout, consider applying
            # later on in the SentenceEncoder (see models.py).
            #  h_lex = self.model.embeddings.dropout(embeddings)
        else:
            lex_seq = None  # dummy; should not be accessed.

        if self.embeddings_mode != "only":
            # encoded_layers is a list of layer activations, each of which is
            # <float32> [batch_size, seq_len, output_dim]
            token_types = self.get_seg_ids(ids)
            _, output_pooled_vec, hidden_states = self.model(
                ids, token_type_ids=token_types, attention_mask=mask
            )

        # <float32> [batch_size, var_seq_len, output_dim]
        return self.prepare_output(lex_seq, hidden_states, mask)


class XLNetEmbedderModule(PytorchTransformersEmbedderModule):
    """ Wrapper for XLNet module to fit into jiant APIs. """

    def __init__(self, args):

        super(XLNetEmbedderModule, self).__init__(args)

        self.model = pytorch_transformers.XLNetModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )

        tokenizer = pytorch_transformers.XLNetTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, do_lower_case="uncased" in args.tokenizer
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.
        self._sep_id = tokenizer.convert_tokens_to_ids("<sep>")
        self._cls_id = tokenizer.convert_tokens_to_ids("<cls>")
        self._pad_id = tokenizer.convert_tokens_to_ids("<pad>")
        self._unk_id = tokenizer.convert_tokens_to_ids("<unk>")

        self.parameter_setup(args)

        # Segment IDs for CLS and SEP tokens. Unlike in BERT, these aren't part of the usual 0/1 input segments.
        # Standard constants reused from pytorch_transformers. They aren't actually used within the pytorch_transformers code, so we're reproducing them here in case they're removed in a later cleanup.
        self._SEG_ID_CLS = 2
        self._SEG_ID_SEP = 3

    @staticmethod
    def apply_boundary_tokens(s1, s2=None):
        # XLNet-style boundary token marking on string token sequences
        if s2:
            return s1 + ["<sep>"] + s2 + ["<sep>", "<cls>"]
        else:
            return s1 + ["<sep>", "<cls>"]

    def forward(
        self, sent: Dict[str, torch.LongTensor], unused_task_name: str = ""
    ) -> torch.FloatTensor:
        """ Run XLNet to get hidden states.

        This forward method does preprocessing on the go,
        changing token IDs from preprocessed word pieces to
        what AllenNLP indexes.

        Args:
            sent: batch dictionary

        Returns:
            h: [batch_size, seq_len, d_emb]
        """
        assert "pytorch_transformers_wpm_pretokenized" in sent

        # <int32> [batch_size, var_seq_len]
        # Make a copy so our padding modifications below don't impact masking decisions elsewhere.
        ids = copy.deepcopy(sent["pytorch_transformers_wpm_pretokenized"])

        mask = ids != 0

        # "Correct" ids to account for different indexing between XLNet and
        # AllenNLP.
        # The AllenNLP indexer adds a '@@UNKNOWN@@' token to the
        # beginning of the vocabulary, *and* treats that as index 1 (index 0 is
        # reserved for native padding).
        ids[ids == 0] = self._pad_id + 2  # Rewrite padding indices.
        ids[ids == 1] = self._unk_id + 2  # Rewrite UNK indices.
        ids -= 2  # shift indices to match XLNet wordpiece embeddings

        if self.embeddings_mode not in ["none", "top"]:
            # This is redundant with the lookup inside XLNetModel,
            # but doing so this way avoids the need to modify the XLNetModel
            # code.
            lex_seq = self.model.word_embedding(ids)
            hidden_states = []  # dummy; should not be accessed.
            # following our use of the OpenAI model, don't use dropout for
            # probing. If you would like to use dropout, consider applying
            # later on in the SentenceEncoder (see models.py).
            #  h_lex = self.model.embeddings.dropout(embeddings)
        else:
            lex_seq = None  # dummy; should not be accessed.

        if self.embeddings_mode != "only":
            # encoded_layers is a list of layer activations, each of which is
            # <float32> [batch_size, seq_len, output_dim]
            token_types = self.get_seg_ids(ids)
            _, output_mems, hidden_states = self.model(
                ids, token_type_ids=token_types, attention_mask=mask
            )

        # <float32> [batch_size, var_seq_len, output_dim]
        return self.prepare_output(lex_seq, hidden_states, mask)
