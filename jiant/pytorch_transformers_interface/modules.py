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

        self.embeddings_mode = args.pytorch_transformers_embedding_mode
        self.input_module = args.input_module
        self.max_seq_len = args.max_seq_len
        self.tokenizer_requird = None

        # Integer token indices for special symbols.
        self._sep_id = None
        self._cls_id = None
        self._pad_id = None
        self._unk_id = None

        # If set, treat these special tokens as part of input segments other than A/B.
        self._SEG_ID_SEP = None
        self._SEG_ID_CLS = None

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
            # Always have one more mixing weight, for lexical layer.
            self.scalar_mix = scalar_mix.ScalarMix(self.max_layer + 1, do_layer_norm=False)

    def correct_sent_indexing(self, sent):
        """ Correct id difference between pytorch_transformers and AllenNLP.
        The AllenNLP indexer adds'@@UNKNOWN@@' token as index 1, and '@@PADDING@@' as index 0
        
        args:
            sent: batch dictionary, in which 
                sent[self.tokenizer_required]: <long> [batch_size, var_seq_len] input token IDs
        
        returns:
            sent["sent_mask"]: <int8> [bath_size, var_seq_len] sentence mask 
            sent["token_id"]: <long> [bath_size, var_seq_len] corrected token IDs
        """
        # import IPython
        # IPython.embed()
        # exit()
        ids = copy.deepcopy(sent[self.tokenizer_requird])

        mask = ids != 0
        ids[ids == 0] = self._pad_id + 2
        # map AllenNLP @@PADDING@@ to _pad_id in specific pytorch_transformer
        if self._unk_id is not None:
            ids[ids == 1] = self._unk_id + 2
            # map AllenNLP @@UNKNOWN@@ to _unk_id in specific pytorch_transformer
        ids -= 2  # shift indices to match pretrained embedding id
        
        sent["token_id"] = ids
        sent["sent_mask"] = mask

    def prepare_output(self, lex_seq, hidden_states, mask):
        """ Compute output according to embedding_mode
        
        args:
            lex_seq: <float32> [batch_size, var_seq_len, hidden_size] lexical embedding
            hidden_states: list of self.max_layer + 1 tensors of
                <float32> [batch_size, var_seq_len, hidden_size],
                all the hidden layers in transformer and the final layer
            mask: <long> [batch_size, var_seq_len] mask of the sentences
        
        returns:
            h: <float32> [batch_size, var_seq_len, output_dim] output embedding
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

        sep_idxs = (token_ids == self._sep_id).long()
        sep_count = torch.cumsum(sep_idxs, dim=-1) - sep_idxs
        seg_ids = (sep_count == 1).long()

        if self._SEG_ID_CLS is not None:
            seg_ids[token_ids == self._cls_id] = self._SEG_ID_CLS

        if self._SEG_ID_SEP is not None:
            seg_ids[token_ids == self._sep_id] = self._SEG_ID_SEP

        return seg_ids

    def forward(self, sent, unused_task_name):
        """ Run transformer model and return output representation 
        
        args:
            sent: batch dictionary, in which 
                sent[self.tokenizer_required]: <long> [batch_size, var_seq_len] input token IDs
            unused_task_name: makeshift input slot, due to an outdated logic in sentence_encoder,
                TODO: deprecate this when sentence_encoder get fixed

        returns:
            transformer_emb: <float32> [batch_size, var_seq_len, output_dim] output embedding
        """
        pass

    def get_pretrained_lm_head(self):
        """ Download another transformer model with LM head, extract the LM head and tie its
        weight to the input token embedding. In most cases, this module needs to work with
        embedding mode "top" or "none"
        
        returns:
            lm_head: module [*, hidden_size] -> [*, vocab_size]
        """
        pass


class BertEmbedderModule(PytorchTransformersEmbedderModule):
    """ Wrapper for BERT module to fit into jiant APIs. """

    def __init__(self, args):
        super(BertEmbedderModule, self).__init__(args)

        self.model = pytorch_transformers.BertModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )

        self.tokenizer_requird = "pytorch_transformers_wpm_pretokenized"
        tokenizer = pytorch_transformers.BertTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, do_lower_case="uncased" in args.tokenizer
        )
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
        
        self.correct_sent_indexing(sent)
        ids = sent["token_id"]
        mask = sent["sent_mask"]

        hidden_states, lex_seq = [], None
        if self.embeddings_mode not in ["none", "top"]:
            lex_seq = self.model.embeddings.word_embeddings(ids)
            lex_seq = self.model.embeddings.LayerNorm(lex_seq)
        if self.embeddings_mode != "only":
            token_types = self.get_seg_ids(ids)
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
    """ Wrapper for XLNet module to fit into jiant APIs. """

    def __init__(self, args):
        super(XLNetEmbedderModule, self).__init__(args)

        self.model = pytorch_transformers.XLNetModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )

        self.tokenizer_requird = "pytorch_transformers_wpm_pretokenized"
        tokenizer = pytorch_transformers.XLNetTokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, do_lower_case="uncased" in args.tokenizer
        )
        self._sep_id = tokenizer.convert_tokens_to_ids("<sep>")
        self._cls_id = tokenizer.convert_tokens_to_ids("<cls>")
        self._pad_id = tokenizer.convert_tokens_to_ids("<pad>")
        self._unk_id = tokenizer.convert_tokens_to_ids("<unk>")

        self.parameter_setup(args)

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
        
        self.correct_sent_indexing(sent)
        ids = sent["token_id"]
        mask = sent["sent_mask"]

        hidden_states, lex_seq = [], None
        if self.embeddings_mode not in ["none", "top"]:
            lex_seq = self.model.word_embedding(ids)
        if self.embeddings_mode != "only":
            token_types = self.get_seg_ids(ids)
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


class TransfoXLEmbedderModule(PytorchTransformersEmbedderModule):
    """ Wrapper for Transformer-XL module to fit into jiant APIs. """

    def __init__(self, args):

        super(TransfoXLEmbedderModule, self).__init__(args)

        self.model = pytorch_transformers.TransfoXLModel.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )

        self.tokenizer_requird = "pytorch_transformers_bytebpe_pretokenized"
        tokenizer = pytorch_transformers.GPT2Tokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )
        self._unk_id = tokenizer.convert_tokens_to_ids("<unk>")
        self._pad_id = tokenizer.convert_tokens_to_ids("<eos>")

        self.parameter_setup(args)

    @staticmethod
    def apply_boundary_tokens(s1):
    # TransformerXL-style boundary token marking on string token sequences
        return s1 + ["<eos>"]
    
    def forward(
        self, sent: Dict[str, torch.LongTensor], unused_task_name: str = ""
    ) -> torch.FloatTensor:

        self.correct_sent_indexing(sent)
        ids = sent["token_id"]
        mask = sent["sent_mask"]

        hidden_states, lex_seq = [], None
        if self.embeddings_mode not in ["none", "top"]:
            lex_seq = self.model.word_emb(ids)
        if self.embeddings_mode != "only":
            _, _, hidden_states = self.model(ids)

    def get_pretrained_lm_head(self):
        model_with_lm_head = pytorch_transformers.TransfoXLLMHeadModel.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.crit
        for i in range(len(model_with_lm_head.crit.out_layers)):
            lm_head.out_layers[i].weight = self.model.word_emb.emb_layers[i].weight
        for i, tie_proj in enumerate(model_with_lm_head.config.tie_projs):
            if tie_proj:
                lm_head.out_projs[i] = self.model.word_emb.emb_projs[i]
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))


class GPT2EmbedderModule(PytorchTransformersEmbedderModule):
    """ Wrapper for GPT2 module to fit into jiant APIs. """

    def __init__(self, args):

        super(GPT2EmbedderModule, self).__init__(args)

        self.model = pytorch_transformers.GPT2Model.from_pretrained(
            args.input_module, cache_dir=self.cache_dir, output_hidden_states=True
        )

        self.tokenizer_requird = "pytorch_transformers_bytebpe_pretokenized"
        tokenizer = pytorch_transformers.GPT2Tokenizer.from_pretrained(
            args.input_module, cache_dir=self.cache_dir
        )
        self._pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        # GPT2 does not have special padding token in its vocab, this is just dummy.

        self.parameter_setup(args)


    @staticmethod
    def apply_boundary_tokens(s1):
    # GPT2-style boundary token marking on string token sequences
        return s1 + ["<|endoftext|>"]

    def forward(
        self, sent: Dict[str, torch.LongTensor], unused_task_name: str = ""
    ) -> torch.FloatTensor:

        self.correct_sent_indexing(sent)
        ids = sent["token_id"]
        mask = sent["sent_mask"]

        hidden_states, lex_seq = [], None
        if self.embeddings_mode not in ["none", "top"]:
            lex_seq = self.model.wte(ids)
        if self.embeddings_mode != "only":
            _, _, hidden_states = self.model(ids)

        return self.prepare_output(lex_seq, hidden_states, mask)

    def get_pretrained_lm_head(self):
        model_with_lm_head = pytorch_transformers.GPT2LMHeadModel.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.lm_head
        lm_head.weight = self.model.wte.weight
        return nn.Sequential(lm_head, nn.LogSoftmax(dim=-1))


class OpenAIGPTEmbedderModule(PytorchTransformersEmbedderModule):
    def __init__(self, args):
        raise NotImplementedError

    @staticmethod
    def apply_boundary_tokens(s1):
        raise NotImplementedError
    
    def forward(
        self, sent: Dict[str, torch.LongTensor], unused_task_name: str = ""
    ) -> torch.FloatTensor:
        raise NotImplementedError

    def get_pretrained_lm_head(self, args):
        raise NotImplementedError