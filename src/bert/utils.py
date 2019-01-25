import logging as log

from typing import Dict

import torch
import torch.nn as nn

# huggingface implementation of BERT
import pytorch_pretrained_bert

class BertEmbedderModule(nn.Module):
    """ Wrapper for BERT module to fit into jiant APIs. """

    def __init__(self, args, cache_dir=None):
        super(BertEmbedderModule, self).__init__()

        self.model = \
            pytorch_pretrained_bert.BertModel.from_pretrained(
                args.bert_model_name,
                cache_dir=cache_dir)
        self.embeddings_mode = args.bert_embeddings_mode
        # BERT supports up to 512 tokens; see section 3.2 of https://arxiv.org/pdf/1810.04805.pdf
        assert args.max_seq_len <= 512
        self.seq_len = args.max_seq_len

        # Set trainability of this module.
        for param in self.model.parameters():
            param.requires_grad = bool(args.bert_fine_tune)

    def forward(self, sent: Dict[str, torch.LongTensor],
                unused_task_name: str="") -> torch.FloatTensor:
        """ Run BERT to get hidden states.

        Args:
            sent: batch dictionary

        Returns:
            h: [batch_size, seq_len, d_emb]
        """
        assert "bert_wpm_pretokenized" in sent
        # <int32> [batch_size, var_seq_len]
        var_ids = sent["bert_wpm_pretokenized"]

        # Model has fixed, learned positional component, so we must pass a
        # block of exactly seq_len length.
        # ids is <int32> [batch_size, seq_len]
        ids = torch.zeros(var_ids.size()[0], self.seq_len, dtype=var_ids.dtype,
                          device=var_ids.device)
        fill_len = min(var_ids.size()[1], self.seq_len)
        ids[:,:fill_len] = var_ids[:,:fill_len]

        mask = (ids != 0)
        # "Correct" ids to account for different indexing between BERT and
        # AllenNLP.
        # The AllenNLP indexer adds a '@@UNKNOWN@@' token to the
        # beginning of the vocabulary, *and* treats that as index 1 (index 0 is
        # reserved for padding).
        FILL_ID = 0  # [PAD] for BERT models.
        ids[ids == 0] = FILL_ID + 2
        # Index 1 should never be used since the BERT WPM uses its own
        # unk token, and handles this at the string level before indexing.
        assert (ids > 1).all()
        ids -= 2

        if self.embeddings_mode != "none":
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
            encoded_layers, _ = self.model(ids, token_type_ids=torch.zeros_like(ids),
                                           attention_mask=mask,
                                           output_all_encoded_layers=True)
            h_enc = encoded_layers[-1]

        if self.embeddings_mode == "none":
            h = h_enc
        elif self.embeddings_mode == "cat":
            h = torch.cat([h_enc, h_lex], dim=2)
        elif self.embeddings_mode == "only":
            h = h_lex
        else:
            raise NotImplementedError(f"embeddings_mode={self.embeddings_mode}"
                                       " not supported.")

        # Truncate back to the original ids length, for compatiblity with the
        # rest of our embedding models. This only drops padding
        # representations.
        h_trunc = h[:,:var_ids.size()[1],:]
        # <float32> [batch_size, var_seq_len, output_dim]
        return h_trunc

    def get_output_dim(self):
        if self.embeddings_mode == "cat":
            return 2*self.model.config.hidden_size
        else:
            return self.model.config.hidden_size


