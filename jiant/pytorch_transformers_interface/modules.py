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
from jiant.pytorch_transformers_interface import input_module_tokenizer_name


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

        self.output_mode = args.pytorch_transformers_output_mode
        self.input_module = args.input_module
        self.max_seq_len = args.max_seq_len
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
        if args.pytorch_transformers_max_layer >= 0:
            self.max_layer = args.pytorch_transformers_max_layer
            assert self.max_layer <= self.num_layers
        else:
            self.max_layer = self.num_layers

        # Configure scalar mixing, ELMo-style.
        if self.output_mode == "mix":
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

    def correct_sent_indexing(self, sent):
        """ Correct id difference between pytorch_transformers and AllenNLP.
        The AllenNLP indexer adds'@@UNKNOWN@@' token as index 1, and '@@PADDING@@' as index 0
        
        args:
            sent: batch dictionary, in which 
                sent[self.tokenizer_required]: <long> [batch_size, var_seq_len] input token IDs
        
        returns:
            ids: <long> [bath_size, var_seq_len] corrected token IDs
            mask: <long> [bath_size, var_seq_len] sentence mask 
        """
        ids = sent[self.tokenizer_required]

        mask = (ids != 0).long()
        ids[ids == 0] = self._pad_id + 2
        # map AllenNLP @@PADDING@@ to _pad_id in specific pytorch_transformer
        if self._unk_id is not None:
            ids[ids == 1] = self._unk_id + 2
            # map AllenNLP @@UNKNOWN@@ to _unk_id in specific pytorch_transformer
        ids -= 2  # shift indexes to match pretrained token embedding indexes

        return ids, mask

    def prepare_output(self, lex_seq, hidden_states, mask):
        """
        Convert the output of the pytorch_transformers module to a vector sequence as expected by jiant.

        args:
            lex_seq: The sequence of input word embeddings as a tensor (batch_size, sequence_length, hidden_size).
                     Used only if output_mode = "only".
            hidden_states: A list of sequences of model hidden states as tensors (batch_size, sequence_length, hidden_size).
            mask: A tensor with 1s in positions corresponding to non-padding tokens (batch_size, sequence_length).

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
            h = self.scalar_mix(available_layers, mask=mask)
        else:
            raise NotImplementedError(f"output_mode={self.output_mode}" " not supported.")

        return h

    def get_output_dim(self):
        if self.output_mode == "cat":
            return 2 * self.model.config.hidden_size
        else:
            return self.model.config.hidden_size

    def get_seg_ids(self, token_ids, mask):
        """ Dynamically build the segment IDs for a concatenated pair of sentences
        Searches for index _sep_id in the tensor. Supports BERT or XLNet-style padding.
        Sets padding tokens to segment zero.

        args:
            token_ids (torch.LongTensor): batch of token IDs
            mask (torch.LongTensor): batch of sentence mask

        returns:
            seg_ids (torch.LongTensor): batch of segment IDs

        example:
        > sents = ["[CLS]", "I", "am", "a", "cat", ".", "[SEP]", "You", "like", "cats", "?", "[SEP]", "[PAD]"]
        > token_tensor = torch.Tensor([[vocab[w] for w in sent]]) # a tensor of token indices
        > seg_ids = get_seg_ids(token_tensor)
        > assert seg_ids == torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        """

        sep_idxs = (token_ids == self._sep_id).long()
        sep_count = torch.cumsum(sep_idxs, dim=-1) - sep_idxs
        seg_ids = sep_count * mask

        if self._SEG_ID_CLS is not None:
            seg_ids[token_ids == self._cls_id] = self._SEG_ID_CLS

        if self._SEG_ID_SEP is not None:
            seg_ids[token_ids == self._sep_id] = self._SEG_ID_SEP

        return seg_ids

    @staticmethod
    def apply_boundary_tokens(s1, s2=None):
        """
        A function that appliese the appropriate EOS/SOS/SEP/CLS tokens to a token sequence.
        
        args:
            s1: list[str], tokens from sentence 1
            s2: list[str] (optional), tokens from sentence 2, used for pair embedding
        
        returns
            s: list[str], token sequence with boundry tokens
        """
        raise NotImplementedError

    def forward(self, sent, unused_task_name):
        """ Run pytorch_transformers model and return output representation 
        
        args:
            sent: batch dictionary, in which 
                sent[self.tokenizer_required]: <long> [batch_size, var_seq_len] input token IDs
            unused_task_name: makeshift input slot, due to some logic in sentence_encoder,
                TODO: deprecate this when sentence_encoder get fixed

        returns:
            transformer_emb: <float32> [batch_size, var_seq_len, output_dim] output embedding
        """
        raise NotImplementedError

    def get_pretrained_lm_head(self):
        """ Download another transformer model with LM head, extract the LM head and tie its
        weight to the input token embedding. In most cases, this module needs to work with
        output_mode as "top" or "none"
        
        returns:
            lm_head: module [*, hidden_size] -> [*, vocab_size]
        """
        raise NotImplementedError


class BertEmbedderModule(PytorchTransformersEmbedderModule):
    """ Wrapper for BERT module to fit into jiant APIs.
    Check PytorchTransformersEmbedderModule for function definitions """

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
        ids, mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.embeddings.word_embeddings(ids)
            lex_seq = self.model.embeddings.LayerNorm(lex_seq)
        if self.output_mode != "only":
            token_types = self.get_seg_ids(ids, mask)
            _, output_pooled_vec, hidden_states = self.model(
                ids, token_type_ids=token_types, attention_mask=mask
            )
        return self.prepare_output(lex_seq, hidden_states, mask)

    def get_pretrained_lm_head(self):
        model_with_lm_head = pytorch_transformers.BertForMaskedLM.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.cls
        lm_head.predictions.decoder.weight = self.model.embeddings.word_embeddings.weight
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))


class XLNetEmbedderModule(PytorchTransformersEmbedderModule):
    """ Wrapper for XLNet module to fit into jiant APIs.
    Check PytorchTransformersEmbedderModule for function definitions """

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

        # Segment IDs for CLS and SEP tokens. Unlike in BERT, these aren't part of the usual 0/1
        # input segments. Standard constants reused from pytorch_transformers. They aren't actually
        # used within the pytorch_transformers code, so we're reproducing them here in case they're
        # removed in a later cleanup.
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
        ids, mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.word_embedding(ids)
        if self.output_mode != "only":
            token_types = self.get_seg_ids(ids, mask)
            _, output_mems, hidden_states = self.model(
                ids, token_type_ids=token_types, attention_mask=mask
            )
        return self.prepare_output(lex_seq, hidden_states, mask)

    def get_pretrained_lm_head(self, args):
        model_with_lm_head = pytorch_transformers.XLNetLMHeadModel.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.lm_loss
        lm_head.weight = self.model.word_embedding.weight
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))


class OpenAIGPTEmbedderModule(PytorchTransformersEmbedderModule):
    """ Wrapper for OpenAIGPT module to fit into jiant APIs.
    Check PytorchTransformersEmbedderModule for function definitions """

    def __init__(self, args):
        super(OpenAIGPTEmbedderModule).__init__(args)

        self.model = pytorch_transformers.OpenAIGPTModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.

        tokenizer = pytorch_transformers.OpenAIGPTTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )
        self._pad_id = tokenizer.convert_tokens_to_ids("\n</w>")
        self._unk_id = tokenizer.convert_tokens_to_ids("<unk>")

        self.parameter_setup(args)

    @staticmethod
    def apply_boundary_tokens(s1):
        # OpenAI-GPT-style boundary token marking on string token sequences
        return ["\n</w>"] + s1 + ["\n</w>"]

    def forward(
        self, sent: Dict[str, torch.LongTensor], unused_task_name: str = ""
    ) -> torch.FloatTensor:
        ids, mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.tokens_embed(ids)
        if self.output_mode != "only":
            _, hidden_states = self.model(ids)
        return self.prepare_output(lex_seq, hidden_states, mask)

    def get_pretrained_lm_head(self, args):
        model_with_lm_head = pytorch_transformers.OpenAIGPTLMHeadModel.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.lm_head
        lm_head.weight = self.model.tokens_embed.weight
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))


class GPT2EmbedderModule(PytorchTransformersEmbedderModule):
    """ Wrapper for GPT-2 module to fit into jiant APIs.
    Check PytorchTransformersEmbedderModule for function definitions """

    def __init__(self, args):
        super(GPT2EmbedderModule, self).__init__(args)

        self.model = pytorch_transformers.GPT2Model.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.

        tokenizer = pytorch_transformers.GPT2Tokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )
        self._pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

        self.parameter_setup(args)

    @staticmethod
    def apply_boundary_tokens(s1):
        # GPT-2-style boundary token marking on string token sequences
        return ["<|endoftext|>"] + s1 + ["<|endoftext|>"]

    def forward(
        self, sent: Dict[str, torch.LongTensor], unused_task_name: str = ""
    ) -> torch.FloatTensor:
        ids, mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.wte(ids)
        if self.output_mode != "only":
            _, _, hidden_states = self.model(ids)
        return self.prepare_output(lex_seq, hidden_states, mask)

    def get_pretrained_lm_head(self):
        model_with_lm_head = pytorch_transformers.GPT2LMHeadModel.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.lm_head
        lm_head.weight = self.model.wte.weight
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))


class TransfoXLEmbedderModule(PytorchTransformersEmbedderModule):
    """ Wrapper for Transformer-XL module to fit into jiant APIs.
    Check PytorchTransformersEmbedderModule for function definitions """

    def __init__(self, args):
        super(TransfoXLEmbedderModule, self).__init__(args)

        self.model = pytorch_transformers.TransfoXLModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.

        tokenizer = pytorch_transformers.TransfoXLTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )
        self._pad_id = tokenizer.convert_tokens_to_ids("<eos>")
        self._unk_id = tokenizer.convert_tokens_to_ids("<unk>")

        self.parameter_setup(args)

    @staticmethod
    def apply_boundary_tokens(s1):
        # TransformerXL-style boundary token marking on string token sequences
        return ["<\n>"] + s1 + ["<\n>"]

    def forward(
        self, sent: Dict[str, torch.LongTensor], unused_task_name: str = ""
    ) -> torch.FloatTensor:
        ids, mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.word_emb(ids)
        if self.output_mode != "only":
            _, _, hidden_states = self.model(ids)
        return self.prepare_output(lex_seq, hidden_states, mask)

    def get_pretrained_lm_head(self):
        # Note: pytorch_transformers didn't implement TransfoXLLMHeadModel, use this in eval only
        model_with_lm_head = pytorch_transformers.TransfoXLLMHeadModel.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.crit
        for i in range(len(model_with_lm_head.crit.out_layers)):
            lm_head.out_layers[i].weight = self.model.word_emb.emb_layers[i].weight
        for i, tie_proj in enumerate(model_with_lm_head.config.tie_projs):
            if tie_proj:
                lm_head.out_projs[i] = self.model.word_emb.emb_projs[i]
        return lm_head


class XLMEmbedderModule(PytorchTransformersEmbedderModule):
    """ Wrapper for XLM module to fit into jiant APIs.
    Check PytorchTransformersEmbedderModule for function definitions """

    def __init__(self, args):
        super(XLMEmbedderModule, self).__init__(args)

        self.model = pytorch_transformers.XLMModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )  # TODO: Speed things up slightly by reusing the previously-loaded tokenizer.

        tokenizer = pytorch_transformers.XLMTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )
        self._unk_id = tokenizer.convert_tokens_to_ids("<unk>")
        self._pad_id = tokenizer.convert_tokens_to_ids("<eos>")

        self.parameter_setup(args)

    @staticmethod
    def apply_boundary_tokens(s1):
        # XLM-style boundary token marking on string token sequences
        return ["</s>"] + s1 + ["</s>"]

    def forward(
        self, sent: Dict[str, torch.LongTensor], unused_task_name: str = ""
    ) -> torch.FloatTensor:
        ids, mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        if self.output_mode not in ["none", "top"]:
            lex_seq = self.model.embeddings(ids)
        if self.output_mode != "only":
            _, _, hidden_states = self.model(ids)
        return self.prepare_output(lex_seq, hidden_states, mask)

    def get_pretrained_lm_head(self):
        model_with_lm_head = pytorch_transformers.XLMWithLMHeadModel.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.pred_layer
        lm_head.proj.weight = self.model.embeddings.weight
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))