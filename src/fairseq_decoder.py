# adapted from https://github.com/pytorch/fairseq/blob/master/fairseq/models/lstm.py
# changes:
#   - deleted everything related to incremental state
#   - add _get_loss to LSTMDecoder (please check in detail)
#   - change decoder initialization to ours

# differences from Seq2SeqDecoder:
#   - very important: instead of mask we're using the padding idx for masking

import torch
import torch.nn as nn
import torch.nn.functional as F

# these four functions are direct copies

def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

def FairseqEmbedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m

def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)
        self.output_proj = Linear(2*output_embed_dim, output_embed_dim, bias=False)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = F.tanh(self.output_proj(torch.cat((x, input), dim=1)))

        return x, attn_scores


class LSTMDecoder(nn.Module):
    """LSTM decoder."""
    def __init__(
        self, num_embeddings, padding_idx, vocab, embed_dim=300, hidden_size=1024, out_embed_dim=1024,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=1024, pretrained_embed=None,
    ):
        super().__init__()
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size

        self.vocab = vocab
        self.padding_idx = padding_idx

        if pretrained_embed is None:
            self.embed_tokens = FairseqEmbedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        assert encoder_output_units == hidden_size, \
            'encoder_output_units ({}) != hidden_size ({})'.format(encoder_output_units, hidden_size)
        # TODO another Linear layer if not equal

        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=encoder_output_units + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        self.attention = AttentionLayer(encoder_output_units, hidden_size) if attention else None
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(self, target_tokens, encoder_outs, encoder_out_mask):
        # very important - prev_tokens vs target_tokens
        # please also check this carefully, as well as _get_loss
        full_target_words = target_tokens["words"]
        prev_output_tokens = full_target_words[:, :-1]
        target_output_tokens = full_target_words[:, 1:]

        bsz, seqlen = prev_output_tokens.size()
        srclen = encoder_outs.size(1)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pass encoder context (our code)
        encoder_padding_mask = 1 - encoder_out_mask.byte().data
        encoder_outs.data.masked_fill_(encoder_padding_mask, -float('inf'))
        decoder_hidden = encoder_outs.new_zeros(encoder_padding_mask.size(0), self.hidden_size)
        decoder_context = encoder_outs.max(dim=1)[0]
        num_layers = len(self.layers)
        prev_hiddens = [decoder_hidden for i in range(num_layers)]
        prev_cells = [decoder_context for i in range(num_layers)]

        # set up encoder outputs for attention
        encoder_outs_a = encoder_outs.clone()
        encoder_outs_a.data.masked_fill_(encoder_padding_mask, -float(0.0))
        encoder_outs_t, encoder_padding_mask_t = encoder_outs_a.transpose(0, 1), encoder_padding_mask.transpose(0, 1).squeeze(-1)

        input_feed = x.data.new(bsz, self.encoder_output_units).zero_()
        attn_scores = x.data.new(srclen, seqlen, bsz).zero_()
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs_t, encoder_padding_mask_t)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of vocabulary
        if hasattr(self, 'additional_fc'):
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)
        x = self.fc_out(x)

        output_dict = {
            "logits": x,
            "loss": self._get_loss(logits=x, target=target_output_tokens),
        }

        return output_dict

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def _get_loss(self, logits, target):
        # adapted from fairseq/criterions/cross_entropy.py
        # extremely important code
        # size_average = True (will not average over ignore_index rows)
        lprobs = self.get_normalized_log_probs(logits)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.contiguous().view(-1,)
        loss_size_average = F.nll_loss(
            lprobs, target, size_average=True, ignore_index=self.padding_idx,
            reduce=True)  # have checked padding_idx
        return loss_size_average

    def get_normalized_log_probs(self, logits):
        """Get normalized log probs from a net's output."""
        return F.log_softmax(logits, dim=-1)
