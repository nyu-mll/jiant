import logging as log
import os
from typing import Dict

import torch
import torch.nn as nn
from allennlp.modules import scalar_mix

import pytorch_transformers

from jiant.preprocess import parse_task_list_arg
from jiant.pytorch_transformers_interface.utils import get_seg_ids
from jiant.utils import utils


class PytorchTransformersEmbedderModule(nn.Module):
    """ Shared code for pytorch_transformers wrappers.

    Subclasses share a good deal of code, but have a number of subtle differences due to different
    APIs from pytorch_transfromers.
    """

    def __init__(self, args):
        super(PytorchTransformersEmbedderModule, self).__init__()

        self.cache_dir = os.getenv(
            "PYTORCH_TRANSFORMERS_CACHE", os.path.join(args.exp_dir, "pytorch_transformers_cache")
        )
        utils.maybe_make_dir(self.cache_dir)

        self.embeddings_mode = args.pytorch_transformers_embedding_mode

    def parameter_setup(self, args):
        # Set trainability of this module.
        for param in self.model.parameters():
            param.requires_grad = bool(args.transfer_paradigm == "finetune")

        # Configure scalar mixing, ELMo-style.
        if self.embeddings_mode == "mix":
            if args.transfer_paradigm == "frozen":
                log.warning(
                    "NOTE: pytorch_transformers_embedding_mode='mix', so scalar "
                    "mixing weights will be fine-tuned even if BERT "
                    "model is frozen."
                )
            # TODO: if doing multiple target tasks, allow for multiple sets of
            # scalars. See the ELMo implementation here:
            # https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L115
            assert len(parse_task_list_arg(args.target_tasks)) <= 1, (
                "pytorch_transformers_embedding_mode='mix' only supports a single set of "
                "scalars (but if you need this feature, see the TODO in "
                "the code!)"
            )
            num_layers = self.model.config.num_hidden_layers
            self.scalar_mix = scalar_mix.ScalarMix(num_layers + 1, do_layer_norm=False)

    def prepare_output(self, output_seq, lex_seq, hidden_states):
        if self.embeddings_mode in ["none", "top"]:
            h = output_seq
        elif self.embeddings_mode == "only":
            h = lex_seq
        elif self.embeddings_mode == "cat":
            h = torch.cat([output_seq, lex_seq], dim=2)
        elif self.embeddings_mode == "mix":
            h = self.scalar_mix([lex_seq] + list(hidden_states[1:]), mask=mask)
        else:
            raise NotImplementedError(f"embeddings_mode={self.embeddings_mode}" " not supported.")

        # <float32> [batch_size, var_seq_len, output_dim]
        return h

    def get_output_dim(self):
        if self.embeddings_mode == "cat":
            return 2 * self.model.config.hidden_size
        else:
            return self.model.config.hidden_size


class BertEmbedderModule(PytorchTransformersEmbedderModule):
    """ Wrapper for BERT module to fit into jiant APIs. """

    def __init__(self, args):
        super(BertEmbedderModule, self).__init__(args)

        self.model = pytorch_transformers.BertModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )

        tokenizer = pytorch_transformers.BertTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.
        self._sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
        self._cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
        self._pad_id = tokenizer.convert_tokens_to_ids("[PAD]")

        self.parameter_setup(args)

    def apply_boundary_tokens(s1, s2=None):
        # BERT-style boundary token padding on string token sequences
        if s2:
            return ["[CLS]"] + s1 + ["[SEP]"] + s2 + ["[SEP]"]
        else:
            return ["[CLS]"] + s1 + ["[SEP]"]

    def forward(
        self, sent: Dict[str, torch.LongTensor], unused_task_name: str = "", is_pair_task=False
    ) -> torch.FloatTensor:
        """ Run BERT to get hidden states.

        This forward method does preprocessing on the go,
        changing token IDs from preprocessed bert to
        what AllenNLP indexes.

        Args:
            sent: batch dictionary
            is_pair_task (bool): true if input is a batch from a pair task

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
            # following our use of the OpenAI model, don't use dropout for
            # probing. If you would like to use dropout, consider applying
            # later on in the SentenceEncoder (see models.py).
            #  h_lex = self.model.embeddings.dropout(embeddings)
        else:
            lex_seq = None

        if self.embeddings_mode != "only":
            # encoded_layers is a list of layer activations, each of which is
            # <float32> [batch_size, seq_len, output_dim]
            token_types = get_seg_ids(ids, self._sep_id) if is_pair_task else torch.zeros_like(ids)
            output_seq, output_pooled_vec, hidden_states = self.model(
                ids, token_type_ids=token_types, attention_mask=mask
            )

        # <float32> [batch_size, var_seq_len, output_dim]
        return self.prepare_output(output_seq, lex_seq, hidden_states)


class XLNetEmbedderModule(PytorchTransformersEmbedderModule):
    """ Wrapper for XLNet module to fit into jiant APIs. """

    def __init__(self, args):

        super(XLNetEmbedderModule, self).__init__(args)

        self.model = pytorch_transformers.XLNetModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )

        tokenizer = pytorch_transformers.XLNetTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.
        self._sep_id = tokenizer.convert_tokens_to_ids("<sep>")
        self._cls_id = tokenizer.convert_tokens_to_ids("<cls>")
        self._pad_id = tokenizer.convert_tokens_to_ids("<pad>")
        print(self._sep_id, self._cls_id, self._pad_id)

        self.parameter_setup(args)

    def apply_boundary_tokens(s1, s2=None):
        # XLNet-style boundary token marking on string token sequences
        if s2:
            return s1 + ["<sep>"] + s2 + ["<sep>", "<cls>"]
        else:
            return s1 + ["<sep>", "<cls>"]

    def forward(
        self, sent: Dict[str, torch.LongTensor], unused_task_name: str = "", is_pair_task=False
    ) -> torch.FloatTensor:
        """ Run XLNet to get hidden states.

        This forward method does preprocessing on the go,
        changing token IDs from preprocessed word pieces to
        what AllenNLP indexes.

        Args:
            sent: batch dictionary
            is_pair_task (bool): true if input is a batch from a pair task

        Returns:
            h: [batch_size, seq_len, d_emb]
        """
        assert "pytorch_transformers_wpm_pretokenized" in sent
        # <int32> [batch_size, var_seq_len]
        ids = sent["pytorch_transformers_wpm_pretokenized"]
        print(ids)
        mask = ids != 0
        # "Correct" ids to account for different indexing between BERT and
        # AllenNLP.
        # The AllenNLP indexer adds a '@@UNKNOWN@@' token to the
        # beginning of the vocabulary, *and* treats that as index 1 (index 0 is
        # reserved for padding).
        ids[ids == 0] = self._pad_id + 2  # Shift the indices that were at 0 to become 2.
        # Index 1 should never be used since the XLNet WPM uses its own
        # unk token, and handles this at the string level before indexing.
        assert (ids > 1).all()
        ids -= 2  # shift indices to match XLNet wordpiece embeddings
        print(ids)

        if self.embeddings_mode not in ["none", "top"]:
            # This is redundant with the lookup inside XLNetModel,
            # but doing so this way avoids the need to modify the BertModel
            # code.
            # Extract lexical embeddings
            lex_seq = self.model.embeddings.word_embeddings(ids)
            lex_seq = self.model.embeddings.LayerNorm(lex_seq)
            # following our use of the OpenAI model, don't use dropout for
            # probing. If you would like to use dropout, consider applying
            # later on in the SentenceEncoder (see models.py).
            #  h_lex = self.model.embeddings.dropout(embeddings)
        else:
            lex_seq = None

        if self.embeddings_mode != "only":
            # encoded_layers is a list of layer activations, each of which is
            # <float32> [batch_size, seq_len, output_dim]
            token_types = get_seg_ids(ids, self._sep_id) if is_pair_task else torch.zeros_like(ids)
            output_seq, output_mems, hidden_states = self.model(
                ids, token_type_ids=token_types, attention_mask=mask
            )

        # <float32> [batch_size, var_seq_len, output_dim]
        return self.prepare_output(output_seq, lex_seq, hidden_states)
