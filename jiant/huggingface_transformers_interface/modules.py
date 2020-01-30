import copy
import logging as log
import os
from typing import Dict

import torch
import torch.nn as nn
from allennlp.modules import scalar_mix

import transformers

from jiant.utils.options import parse_task_list_arg
from jiant.utils import utils
from jiant.huggingface_transformers_interface import input_module_tokenizer_name


class HuggingfaceTransformersEmbedderModule(nn.Module):
    """ Shared code for transformers wrappers.

    Subclasses share a good deal of code, but have a number of subtle differences due to different
    APIs from transfromers.
    """

    def __init__(self, args):
        super(HuggingfaceTransformersEmbedderModule, self).__init__()

        self.cache_dir = os.getenv(
            "HUGGINGFACE_TRANSFORMERS_CACHE", os.path.join(args.exp_dir, "transformers_cache")
        )
        utils.maybe_make_dir(self.cache_dir)

        self.output_mode = args.transformers_output_mode
        self.input_module = args.input_module
        self.max_pos = None
        self.tokenizer_required = input_module_tokenizer_name(args.input_module)

        # Integer token indices for special symbols.
        self._cls_id = None
        self._sep_id = None
        self._pad_id = None
        self._unk_id = None

        # If set, treat these special tokens as part of input segments other than A/B.
        self._SEG_ID_CLS = None
        self._SEG_ID_SEP = None

    def parameter_setup(self, args):
        # Set trainability of this module.
        for param in self.model.parameters():
            param.requires_grad = bool(args.transfer_paradigm == "finetune")

        self.num_layers = self.model.config.num_hidden_layers
        if args.transformers_max_layer >= 0:
            self.max_layer = args.transformers_max_layer
            assert self.max_layer <= self.num_layers
        else:
            self.max_layer = self.num_layers

        if args.transfer_paradigm == "frozen":
            if isinstance(
                self, (OpenAIGPTEmbedderModule, GPT2EmbedderModule, TransfoXLEmbedderModule)
            ):
                log.warning(
                    "NOTE: OpenAI GPT, GPT-2 and Transformer-XL add new tokens for classification"
                    "tasks, under 'frozen' transfer_paradigm, their embeddings will not be trained"
                )

        # Configure scalar mixing, ELMo-style.
        if self.output_mode == "mix":
            if args.transfer_paradigm == "frozen":
                log.warning(
                    "NOTE: transformers_output_mode='mix', so scalar "
                    "mixing weights will be fine-tuned even if BERT "
                    "model is frozen."
                )
            # TODO: if doing multiple target tasks, allow for multiple sets of
            # scalars. See the ELMo implementation here:
            # https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L115
            assert len(parse_task_list_arg(args.target_tasks)) <= 1, (
                "transformers_output_mode='mix' only supports a single set of "
                "scalars (but if you need this feature, see the TODO in "
                "the code!)"
            )
            # Always have one more mixing weight, for lexical layer.
            self.scalar_mix = scalar_mix.ScalarMix(self.max_layer + 1, do_layer_norm=False)

    def correct_sent_indexing(self, sent):
        """ Correct id difference between transformers and AllenNLP.
        The AllenNLP indexer adds'@@UNKNOWN@@' token as index 1, and '@@PADDING@@' as index 0

        args:
            sent: batch dictionary, in which
                sent[self.tokenizer_required]: <long> [batch_size, var_seq_len] input token IDs

        returns:
            ids: <long> [bath_size, var_seq_len] corrected token IDs
            input_mask: <long> [bath_size, var_seq_len] mask of input sequence
        """
        assert (
            self.tokenizer_required in sent
        ), "transformers cannot find correcpondingly tokenized input"
        ids = sent[self.tokenizer_required]

        input_mask = (ids != 0).long()
        pad_mask = (ids == 0).long()
        # map AllenNLP @@PADDING@@ to _pad_id in specific transformer vocab
        unk_mask = (ids == 1).long()
        # map AllenNLP @@UNKNOWN@@ to _unk_id in specific transformer vocab
        valid_mask = (ids > 1).long()
        # shift ordinary indexes by 2 to match pretrained token embedding indexes
        if self._unk_id is not None:
            ids = (ids - 2) * valid_mask + self._pad_id * pad_mask + self._unk_id * unk_mask
        else:
            ids = (ids - 2) * valid_mask + self._pad_id * pad_mask
            assert (
                unk_mask == 0
            ).all(), "out-of-vocabulary token found in the input, but _unk_id of transformers model is not specified"
        if self.max_pos is not None:
            assert (
                ids.size()[-1] <= self.max_pos
            ), "input length exceeds position embedding capacity, reduce max_seq_len"

        sent[self.tokenizer_required] = ids
        return ids, input_mask

    def prepare_output(self, lex_seq, hidden_states, input_mask):
        """
        Convert the output of the transformers module to a vector sequence as expected by jiant.

        args:
            lex_seq: The sequence of input word embeddings as a tensor (batch_size, sequence_length, hidden_size).
                     Used only if output_mode = "only".
            hidden_states: A list of sequences of model hidden states as tensors (batch_size, sequence_length, hidden_size).
            input_mask: A tensor with 1s in positions corresponding to non-padding tokens (batch_size, sequence_length).

        returns:
            h: Output embedding as a tensor (batch_size, sequence_length, output_dim)
        """
        available_layers = hidden_states[: self.max_layer + 1]

        if self.output_mode in ["none", "top"]:
            h = available_layers[-1]
        elif self.output_mode == "only":
            h = lex_seq
        elif self.output_mode == "cat":
            h = torch.cat([available_layers[-1], lex_seq], dim=2)
        elif self.output_mode == "mix":
            h = self.scalar_mix(available_layers, mask=input_mask)
        else:
            raise NotImplementedError(f"output_mode={self.output_mode}" " not supported.")

        return h

    def get_output_dim(self):
        if self.output_mode == "cat":
            return 2 * self.model.config.hidden_size
        else:
            return self.model.config.hidden_size

    def get_seg_ids(self, token_ids, input_mask):
        """ Dynamically build the segment IDs for a concatenated pair of sentences
        Searches for index _sep_id in the tensor. Supports BERT or XLNet-style padding.
        Sets padding tokens to segment zero.

        args:
            token_ids (torch.LongTensor): batch of token IDs
            input_mask (torch.LongTensor): mask of token_ids

        returns:
            seg_ids (torch.LongTensor): batch of segment IDs

        example:
        > sents = ["[CLS]", "I", "am", "a", "cat", ".", "[SEP]", "You", "like", "cats", "?", "[SEP]", "[PAD]"]
        > token_tensor = torch.Tensor([[vocab[w] for w in sent]]) # a tensor of token indices
        > seg_ids = get_seg_ids(token_tensor, torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]))
        > assert seg_ids == torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        """
        # TODO: creating sentence segment id(and language segment id for XLM) is more suitable for preprocess
        sep_idxs = (token_ids == self._sep_id).long()
        sep_count = torch.cumsum(sep_idxs, dim=-1) - sep_idxs
        seg_ids = sep_count * input_mask

        if self._SEG_ID_CLS is not None:
            seg_ids[token_ids == self._cls_id] = self._SEG_ID_CLS

        if self._SEG_ID_SEP is not None:
            seg_ids[token_ids == self._sep_id] = self._SEG_ID_SEP

        return seg_ids

    @staticmethod
    def apply_boundary_tokens(s1, s2=None, get_offset=False):
        """
        A function that appliese the appropriate EOS/SOS/SEP/CLS tokens to token sequence or
        token sequence pair for most tasks.
        This function should be implmented in subclasses.

        args:
            s1: list[str], tokens from sentence 1
            s2: list[str] (optional), tokens from sentence 2, used for pair embedding
            get_offset: bool, returns offset if True

        returns
            s: list[str], token sequence with boundry tokens
            offset_s1 (optional): int, index offset of s1
            offset_s2 (optional): int, index offset of s2
        """
        raise NotImplementedError

    @staticmethod
    def apply_lm_boundary_tokens(s1, get_offset=False):
        """
        A function that appliese the appropriate EOS/SOS/SEP/CLS tokens to a token sequence for
        language modeling tasks.
        This function should be implmented in subclasses.

        args:
            s1: list[str], tokens from sentence
            get_offset: bool, returns offset if True

        returns
            s: list[str], token sequence with boundry tokens
            offset_s1 (optional): int, index offset of s1
        """
        raise NotImplementedError

    def forward(self, sent, task_name):
        """ Run transformers model and return output representation
        This function should be implmented in subclasses.

        args:
            sent: batch dictionary, in which
                sent[self.tokenizer_required]: <long> [batch_size, var_seq_len] input token IDs
            task_name: task_name string, this can used to implement different mixing scalars for
                differnt tasks. See the TODO in parameter_setup for more details.

        returns:
            transformer_emb: <float32> [batch_size, var_seq_len, output_dim] output embedding
        """
        raise NotImplementedError

    def get_pretrained_lm_head(self):
        """ Download another transformer model with LM head, extract the LM head and tie its
        weight to the input token embedding. In most cases, this module needs to work with
        output_mode as "top" or "none"
        This function should be implmented in subclasses.

        returns:
            lm_head: module [*, hidden_size] -> [*, vocab_size]
        """
        raise NotImplementedError


class BertEmbedderModule(HuggingfaceTransformersEmbedderModule):
    """ Wrapper for BERT module to fit into jiant APIs.
    Check HuggingfaceTransformersEmbedderModule for function definitions """

    def __init__(self, args):
        super(BertEmbedderModule, self).__init__(args)

        self.model = transformers.BertModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )
        self.max_pos = self.model.config.max_position_embeddings

        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, do_lower_case="uncased" in args.tokenizer
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.
        self._sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self._cls_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self._pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self._unk_id = self.tokenizer.convert_tokens_to_ids("[UNK]")

        self.parameter_setup(args)

    @staticmethod
    def apply_boundary_tokens(s1, s2=None, get_offset=False):
        # BERT-style boundary token padding on string token sequences
        if s2:
            s = ["[CLS]"] + s1 + ["[SEP]"] + s2 + ["[SEP]"]
            if get_offset:
                return s, 1, len(s1) + 2
        else:
            s = ["[CLS]"] + s1 + ["[SEP]"]
            if get_offset:
                return s, 1
        return s

    def forward(self, sent: Dict[str, torch.LongTensor], task_name: str = "") -> torch.FloatTensor:
        ids, input_mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.embeddings.word_embeddings(ids)
            lex_seq = self.model.embeddings.LayerNorm(lex_seq)
        if self.output_mode != "only":
            token_types = self.get_seg_ids(ids, input_mask)
            _, output_pooled_vec, hidden_states = self.model(
                ids, token_type_ids=token_types, attention_mask=input_mask
            )
        return self.prepare_output(lex_seq, hidden_states, input_mask)

    def get_pretrained_lm_head(self):
        model_with_lm_head = transformers.BertForMaskedLM.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.cls
        lm_head.predictions.decoder.weight = self.model.embeddings.word_embeddings.weight
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))


class RobertaEmbedderModule(HuggingfaceTransformersEmbedderModule):
    """ Wrapper for RoBERTa module to fit into jiant APIs.
    Check HuggingfaceTransformersEmbedderModule for function definitions """

    def __init__(self, args):
        super(RobertaEmbedderModule, self).__init__(args)

        self.model = transformers.RobertaModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )
        self.max_pos = self.model.config.max_position_embeddings

        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.
        self._sep_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self._cls_id = self.tokenizer.convert_tokens_to_ids("<s>")
        self._pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        self._unk_id = self.tokenizer.convert_tokens_to_ids("<unk>")

        self.parameter_setup(args)

    @staticmethod
    def apply_boundary_tokens(s1, s2=None, get_offset=False):
        # RoBERTa-style boundary token padding on string token sequences
        if s2:
            s = ["<s>"] + s1 + ["</s>", "</s>"] + s2 + ["</s>"]
            if get_offset:
                return s, 1, len(s1) + 3
        else:
            s = ["<s>"] + s1 + ["</s>"]
            if get_offset:
                return s, 1
        return s

    def forward(self, sent: Dict[str, torch.LongTensor], task_name: str = "") -> torch.FloatTensor:
        ids, input_mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.embeddings.word_embeddings(ids)
            lex_seq = self.model.embeddings.LayerNorm(lex_seq)
        if self.output_mode != "only":
            _, output_pooled_vec, hidden_states = self.model(ids, attention_mask=input_mask)
        return self.prepare_output(lex_seq, hidden_states, input_mask)

    def get_pretrained_lm_head(self):
        model_with_lm_head = transformers.RobertaForMaskedLM.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.lm_head
        lm_head.predictions.decoder.weight = self.model.embeddings.word_embeddings.weight
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))


class AlbertEmbedderModule(HuggingfaceTransformersEmbedderModule):
    """ Wrapper for ALBERT module to fit into jiant APIs.
    Check HuggingfaceTransformersEmbedderModule for function definitions """

    def __init__(self, args):
        super(AlbertEmbedderModule, self).__init__(args)

        self.model = transformers.AlbertModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )
        self.max_pos = self.model.config.max_position_embeddings

        self.tokenizer = transformers.AlbertTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.
        self._sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self._cls_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self._pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        self._unk_id = self.tokenizer.convert_tokens_to_ids("<unk>")

        self.parameter_setup(args)

    @staticmethod
    def apply_boundary_tokens(s1, s2=None, get_offset=False):
        # ALBERT-style boundary token padding on string token sequences
        if s2:
            s = ["[CLS]"] + s1 + ["[SEP]"] + s2 + ["[SEP]"]
            if get_offset:
                return s, 1, len(s1) + 2
        else:
            s = ["[CLS]"] + s1 + ["[SEP]"]
            if get_offset:
                return s, 1
        return s

    def forward(self, sent: Dict[str, torch.LongTensor], task_name: str = "") -> torch.FloatTensor:
        ids, input_mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.embeddings.word_embeddings(ids)
            lex_seq = self.model.embeddings.LayerNorm(lex_seq)
        if self.output_mode != "only":
            token_types = self.get_seg_ids(ids, input_mask)
            _, output_pooled_vec, hidden_states = self.model(
                ids, token_type_ids=token_types, attention_mask=input_mask
            )
        return self.prepare_output(lex_seq, hidden_states, input_mask)

    def get_pretrained_lm_head(self):
        model_with_lm_head = transformers.AlbertForMaskedLM.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.predictions
        lm_head.decoder.weight = self.model.embeddings.word_embeddings.weight
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))


class XLNetEmbedderModule(HuggingfaceTransformersEmbedderModule):
    """ Wrapper for XLNet module to fit into jiant APIs.
    Check HuggingfaceTransformersEmbedderModule for function definitions """

    def __init__(self, args):
        super(XLNetEmbedderModule, self).__init__(args)

        self.model = transformers.XLNetModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )

        self.tokenizer = transformers.XLNetTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, do_lower_case="uncased" in args.tokenizer
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.
        self._sep_id = self.tokenizer.convert_tokens_to_ids("<sep>")
        self._cls_id = self.tokenizer.convert_tokens_to_ids("<cls>")
        self._pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        self._unk_id = self.tokenizer.convert_tokens_to_ids("<unk>")

        self.parameter_setup(args)

        # Segment IDs for CLS and SEP tokens. Unlike in BERT, these aren't part of the usual 0/1
        # input segments. Standard constants reused from transformers. They aren't actually
        # used within the transformers code, so we're reproducing them here in case they're
        # removed in a later cleanup.
        self._SEG_ID_CLS = 2
        self._SEG_ID_SEP = 3

    @staticmethod
    def apply_boundary_tokens(s1, s2=None, get_offset=False):
        # XLNet-style boundary token marking on string token sequences
        if s2:
            s = s1 + ["<sep>"] + s2 + ["<sep>", "<cls>"]
            if get_offset:
                return s, 0, len(s1) + 1
        else:
            s = s1 + ["<sep>", "<cls>"]
            if get_offset:
                return s, 0
        return s

    def forward(self, sent: Dict[str, torch.LongTensor], task_name: str = "") -> torch.FloatTensor:
        ids, input_mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.word_embedding(ids)
        if self.output_mode != "only":
            token_types = self.get_seg_ids(ids, input_mask)
            _, output_mems, hidden_states = self.model(
                ids, token_type_ids=token_types, attention_mask=input_mask
            )
        return self.prepare_output(lex_seq, hidden_states, input_mask)

    def get_pretrained_lm_head(self, args):
        model_with_lm_head = transformers.XLNetLMHeadModel.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.lm_loss
        lm_head.weight = self.model.word_embedding.weight
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))


class OpenAIGPTEmbedderModule(HuggingfaceTransformersEmbedderModule):
    """ Wrapper for OpenAI GPT module to fit into jiant APIs.
    Check HuggingfaceTransformersEmbedderModule for function definitions """

    def __init__(self, args):
        super(OpenAIGPTEmbedderModule, self).__init__(args)

        self.model = transformers.OpenAIGPTModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.
        self.max_pos = self.model.config.n_positions

        self.tokenizer = transformers.OpenAIGPTTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )
        self._pad_id = self.tokenizer.convert_tokens_to_ids("\n</w>")
        self._unk_id = self.tokenizer.convert_tokens_to_ids("<unk>")

        special_tokens = {"bos_token": "<start>", "sep_token": "<delim>", "cls_token": "<extract>"}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.parameter_setup(args)

    @staticmethod
    def apply_boundary_tokens(s1, s2=None, get_offset=False):
        # OpenAI-GPT-style boundary token marking on string token sequences
        if s2:
            s = ["<start>"] + s1 + ["<delim>"] + s2 + ["<extract>"]
            if get_offset:
                return s, 1, len(s1) + 2
        else:
            s = ["<start>"] + s1 + ["<extract>"]
            if get_offset:
                return s, 1
        return s

    @staticmethod
    def apply_lm_boundary_tokens(s1, get_offset=False):
        # OpenAI-GPT-style boundary token marking on string token sequences for LM tasks
        s = ["\n</w>"] + s1 + ["\n</w>"]
        if get_offset:
            return s, 1
        return s

    def forward(self, sent: Dict[str, torch.LongTensor], task_name: str = "") -> torch.FloatTensor:
        ids, input_mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.tokens_embed(ids)
        if self.output_mode != "only":
            _, hidden_states = self.model(ids)
        return self.prepare_output(lex_seq, hidden_states, input_mask)

    def get_pretrained_lm_head(self, args):
        model_with_lm_head = transformers.OpenAIGPTLMHeadModel.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.lm_head
        lm_head.weight = self.model.tokens_embed.weight[: lm_head.weight.size()[0]]
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))


class GPT2EmbedderModule(HuggingfaceTransformersEmbedderModule):
    """ Wrapper for GPT-2 module to fit into jiant APIs.
    Check HuggingfaceTransformersEmbedderModule for function definitions """

    def __init__(self, args):
        super(GPT2EmbedderModule, self).__init__(args)

        self.model = transformers.GPT2Model.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.
        self.max_pos = self.model.config.n_positions

        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )
        self._pad_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

        special_tokens = {"bos_token": "<start>", "sep_token": "<delim>", "cls_token": "<extract>"}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.parameter_setup(args)

    @staticmethod
    def apply_boundary_tokens(s1, s2=None, get_offset=False):
        # GPT-2-style boundary token marking on string token sequences
        if s2:
            s = ["<start>"] + s1 + ["<delim>"] + s2 + ["<extract>"]
            if get_offset:
                return s, 1, len(s1) + 2
        else:
            s = ["<start>"] + s1 + ["<extract>"]
            if get_offset:
                return s, 1
        return s

    @staticmethod
    def apply_lm_boundary_tokens(s1, get_offset=False):
        # GPT-2-style boundary token marking on string token sequences for LM tasks
        s = ["<|endoftext|>"] + s1 + ["<|endoftext|>"]
        if get_offset:
            return s, 1
        return s

    def forward(self, sent: Dict[str, torch.LongTensor], task_name: str = "") -> torch.FloatTensor:
        ids, input_mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.wte(ids)
        if self.output_mode != "only":
            _, _, hidden_states = self.model(ids)
        return self.prepare_output(lex_seq, hidden_states, input_mask)

    def get_pretrained_lm_head(self):
        model_with_lm_head = transformers.GPT2LMHeadModel.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.lm_head
        lm_head.weight = self.model.wte.weight[: lm_head.weight.size()[0]]
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))


class TransfoXLEmbedderModule(HuggingfaceTransformersEmbedderModule):
    """ Wrapper for Transformer-XL module to fit into jiant APIs.
    Check HuggingfaceTransformersEmbedderModule for function definitions """

    def __init__(self, args):
        super(TransfoXLEmbedderModule, self).__init__(args)

        self.model = transformers.TransfoXLModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.

        self.tokenizer = transformers.TransfoXLTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )
        self._pad_id = self.tokenizer.convert_tokens_to_ids("<eos>")
        self._unk_id = self.tokenizer.convert_tokens_to_ids("<unk>")

        special_tokens = {"bos_token": "<start>", "sep_token": "<delim>", "cls_token": "<extract>"}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.parameter_setup(args)

    @staticmethod
    def apply_boundary_tokens(s1, s2=None, get_offset=False):
        # TransformerXL-style boundary token marking on string token sequences
        if s2:
            s = ["<start>"] + s1 + ["<delim>"] + s2 + ["<extract>"]
            if get_offset:
                return s, 1, len(s1) + 2
        else:
            s = ["<start>"] + s1 + ["<extract>"]
            if get_offset:
                return s, 1
        return s

    @staticmethod
    def apply_lm_boundary_tokens(s1, get_offset=False):
        # TransformerXL-style boundary token marking on string token sequences for LM tasks
        s = ["<\n>"] + s1 + ["<\n>"]
        if get_offset:
            return s, 1
        return s

    def forward(self, sent: Dict[str, torch.LongTensor], task_name: str = "") -> torch.FloatTensor:
        ids, input_mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.word_emb(ids)
        if self.output_mode != "only":
            _, _, hidden_states = self.model(ids)
        return self.prepare_output(lex_seq, hidden_states, input_mask)

    def get_pretrained_lm_head(self):
        # Note: transformers didn't implement TransfoXLLMHeadModel, use this in eval only
        model_with_lm_head = transformers.TransfoXLLMHeadModel.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.crit
        for i in range(len(model_with_lm_head.crit.out_layers) - 1):
            lm_head.out_layers[i].weight = self.model.word_emb.emb_layers[i].weights
        lm_head.out_layers[-1].weight = self.model.word_emb.emb_layers[-1].weight[
            : lm_head.out_layers[-1].weight.size()[0]
        ]
        for i, tie_proj in enumerate(model_with_lm_head.config.tie_projs):
            if tie_proj:
                lm_head.out_projs[i] = self.model.word_emb.emb_projs[i]
        return lm_head


class XLMEmbedderModule(HuggingfaceTransformersEmbedderModule):
    """ Wrapper for XLM module to fit into jiant APIs.
    Check HuggingfaceTransformersEmbedderModule for function definitions """

    def __init__(self, args):
        super(XLMEmbedderModule, self).__init__(args)

        self.model = transformers.XLMModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.
        self.max_pos = self.model.config.max_position_embeddings

        self.tokenizer = transformers.XLMTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )
        self._unk_id = self.tokenizer.convert_tokens_to_ids("<unk>")
        self._pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")

        self.parameter_setup(args)

    @staticmethod
    def apply_boundary_tokens(s1, s2=None, get_offset=False):
        # XLM-style boundary token marking on string token sequences
        if s2:
            s = ["</s>"] + s1 + ["</s>"] + s2 + ["</s>"]
            if get_offset:
                return s, 1, len(s1) + 2
        else:
            s = ["</s>"] + s1 + ["</s>"]
            if get_offset:
                return s, 1, len(s1) + 1
        return s

    def forward(self, sent: Dict[str, torch.LongTensor], task_name: str = "") -> torch.FloatTensor:
        ids, input_mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.embeddings(ids)
        if self.output_mode != "only":
            _, hidden_states = self.model(ids)
        return self.prepare_output(lex_seq, hidden_states, input_mask)

    def get_pretrained_lm_head(self):
        model_with_lm_head = transformers.XLMWithLMHeadModel.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.pred_layer
        lm_head.proj.weight = self.model.embeddings.weight
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))
