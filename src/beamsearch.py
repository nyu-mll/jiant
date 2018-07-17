
import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from torch.autograd import Variable
import torch.nn.functional as F


BEAM_SIZE = 3


"""Beam search implementation in PyTorch."""
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

# Code borrowed from PyTorch OpenNMT example
# https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Beam.py


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, vocab, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.vocab = vocab

        self.pad = self.vocab.get_token_index(START_SYMBOL, "targets")  # TODO
        self.bos = self.vocab.get_token_index(START_SYMBOL, "targets")
        self.eos = self.vocab.get_token_index(END_SYMBOL, "targets")
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[0], ids[0]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]


def beam_search(decoder, encoder_outputs, encoder_mask, beam_size=BEAM_SIZE):
    """ performs beam search. returns hypotheses with scores."""
    batch_size = encoder_outputs.size(0)
    encoder_hidden_dim = encoder_outputs.size(2)
    assert encoder_hidden_dim == decoder._decoder_hidden_dim
    trg_h_t = encoder_outputs.max(dim=1)[0]
    trg_c_t = encoder_outputs.new_zeros(encoder_outputs.size(0), decoder._decoder_hidden_dim)  # [bs, 1, h]

    max_trg_length = decoder._max_decoding_steps

    # Expand tensors for each beam.
    dec_states = [
        Variable(trg_h_t.data.repeat(beam_size, 1)),  # [bs*beam_size, h]
        Variable(trg_c_t.data.repeat(beam_size, 1))
    ]

    beam = [
        Beam(beam_size, decoder.vocab, cuda=True)
        for k in range(batch_size)
    ]

    batch_idx = list(range(batch_size))
    remaining_sents = batch_size

    for _ in range(max_trg_length):
        input_indices = torch.stack(
            [b.get_current_state() for b in beam if not b.done]
        ).contiguous().view(-1)  # [beam_size*bs]
        decoder_input = decoder._prepare_decode_step_input(input_indices)

        logits, trg_h_t, trg_c_t = decoder._decoder_step(
            decoder_input,
            dec_states[0],
            dec_states[1],
        )

        dec_states = (trg_h_t, trg_c_t)
        dec_out = trg_h_t.squeeze(1)

        word_lk = F.softmax(logits, dim=1).view(  # (softmax is not really necessary)
            beam_size,
            remaining_sents,
            -1
        ).transpose(0, 1).contiguous()

        active = []
        for b in range(batch_size):
            if beam[b].done:
                continue

            idx = batch_idx[b]
            if not beam[b].advance(word_lk.data[idx]):
                active += [b]

        if not active:
            break

        # in this section, the sentences that are still active are
        # compacted so that the decoder is not run on completed sentences
        active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
        batch_idx = {beam: idx for idx, beam in enumerate(active)}

        def update_active(t):
            # select only the remaining active sentences
            view = t.data.view(
                -1, remaining_sents,
                decoder._decoder_hidden_dim
            )
            new_size = list(t.size())
            new_size[-2] = new_size[-2] * len(active_idx) \
                // remaining_sents
            return Variable(view.index_select(
                1, active_idx
            ).view(*new_size))

        dec_states = (
            update_active(dec_states[0]),
            update_active(dec_states[1])
        )
        dec_out = update_active(dec_out)

        remaining_sents = len(active)

    #  (4) package everything up

    allHyp, allScores = [], []
    n_best = 1

    for b in range(batch_size):
        scores, ks = beam[b].sort_best()

        allScores += [scores[:n_best]]
        hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
        allHyp += [hyps]

    return allHyp, allScores


def compute_bleu(hyps, scores, targets):
    pass


def generate_and_compute_bleu(decoder, encoder_outputs, encoder_mask, targets):
    hyps, scores = beam_search(decoder, encoder_outputs, encoder_mask)
    bleu_score = compute_bleu(hyps, scores, targets)

    return bleu_score
