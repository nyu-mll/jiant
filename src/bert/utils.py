import logging as log

from typing import Dict

import torch
import torch.nn as nn

from ..preprocess import parse_task_list_arg

from allennlp.modules import scalar_mix

# huggingface implementation of BERT
import pytorch_pretrained_bert


def _get_seg_ids(ids, sep_id):
    """ Dynamically build the segment IDs for a concatenated pair of sentences
    Searches for index SEP_ID in the tensor

    args:
        ids (torch.LongTensor): batch of token IDs

    returns:
        seg_ids (torch.LongTensor): batch of segment IDs

    example:
    > sents = ["[CLS]", "I", "am", "a", "cat", ".", "[SEP]", "You", "like", "cats", "?", "[SEP]"]
    > token_tensor = torch.Tensor([[vocab[w] for w in sent]]) # a tensor of token indices
    > seg_ids = _get_seg_ids(token_tensor, sep_id=102) # BERT [SEP] ID
    > assert seg_ids == torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    """
    sep_idxs = (ids == sep_id).nonzero()[:, 1]
    seg_ids = torch.ones_like(ids)
    for row, idx in zip(seg_ids, sep_idxs[::2]):
        row[:idx + 1].fill_(0)
    return seg_ids


class BertEmbedderModule(nn.Module):
    """ Wrapper for BERT module to fit into jiant APIs. """

    def __init__(self, args, cache_dir=None):
        super(BertEmbedderModule, self).__init__()

        self.model = \
            pytorch_pretrained_bert.BertModel.from_pretrained(
                args.bert_model_name,
                cache_dir=cache_dir)
        self.embeddings_mode = args.bert_embeddings_mode

        tokenizer = \
            pytorch_pretrained_bert.BertTokenizer.from_pretrained(
                args.bert_model_name,
                cache_dir=cache_dir)
        self._sep_id = tokenizer.vocab["[SEP]"]
        self._pad_id = tokenizer.vocab["[PAD]"]

        # Set trainability of this module.
        for param in self.model.parameters():
            param.requires_grad = bool(args.transfer_paradigm == 'finetune')

        # Configure scalar mixing, ELMo-style.
        if self.embeddings_mode == "mix":
            if args.transfer_paradigm == 'frozen':
                log.warning("NOTE: bert_embeddings_mode='mix', so scalar "
                            "mixing weights will be fine-tuned even if BERT "
                            "model is frozen.")
            # TODO: if doing multiple target tasks, allow for multiple sets of
            # scalars. See the ELMo implementation here:
            # https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L115
            assert len(parse_task_list_arg(args.target_tasks)) <= 1, \
                ("bert_embeddings_mode='mix' only supports a single set of "
                 "scalars (but if you need this feature, see the TODO in "
                 "the code!)")
            num_layers = self.model.config.num_hidden_layers
            self.scalar_mix = scalar_mix.ScalarMix(num_layers + 1,
                                                   do_layer_norm=False)

    def forward(self, sent: Dict[str, torch.LongTensor],
                unused_task_name: str="",
                is_pair_task=False) -> torch.FloatTensor:
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
        assert "bert_wpm_pretokenized" in sent
        # <int32> [batch_size, var_seq_len]
        ids = sent["bert_wpm_pretokenized"]
        # BERT supports up to 512 tokens; see section 3.2 of https://arxiv.org/pdf/1810.04805.pdf
        assert ids.size()[1] <= 512

        mask = (ids != 0)
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
            # Extract lexical embeddings; see
            # https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L186
            h_lex = self.model.embeddings.word_embeddings(ids)
            h_lex = self.model.embeddings.LayerNorm(h_lex)
            # following our use of the OpenAI model, don't use dropout for
            # probing. If you would like to use dropout, consider applying
            # later on in the SentenceEncoder (see models.py).
            #  h_lex = self.model.embeddings.dropout(embeddings)

        if self.embeddings_mode != "only":
            # encoded_layers is a list of layer activations, each of which is
            # <float32> [batch_size, seq_len, output_dim]
            token_types = _get_seg_ids(ids, self._sep_id) if is_pair_task else torch.zeros_like(ids)
            encoded_layers, _ = self.model(ids, token_type_ids=token_types,
                                           attention_mask=mask,
                                           output_all_encoded_layers=True)
            h_enc = encoded_layers[-1]

        if self.embeddings_mode in ["none", "top"]:
            h = h_enc
        elif self.embeddings_mode == "only":
            h = h_lex
        elif self.embeddings_mode == "cat":
            h = torch.cat([h_enc, h_lex], dim=2)
        elif self.embeddings_mode == "mix":
            h = self.scalar_mix([h_lex] + encoded_layers, mask=mask)
        else:
            raise NotImplementedError(f"embeddings_mode={self.embeddings_mode}"
                                      " not supported.")

        # <float32> [batch_size, var_seq_len, output_dim]
        return h

    def get_output_dim(self):
        if self.embeddings_mode == "cat":
            return 2 * self.model.config.hidden_size
        else:
            return self.model.config.hidden_size
