import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from torch.autograd import Variable
import torch.nn.functional as F

from . import bleu_scoring

BEAM_SIZE = 3


"""Beam search implementation in PyTorch."""
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

        self.bos = self.vocab.get_token_index(START_SYMBOL, "targets")
        self.eos = self.vocab.get_token_index(END_SYMBOL, "targets")
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.bos)]
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
    # Returns the top hypothesis
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            word_idx = int(self.nextYs[j + 1][k].item())
            if word_idx != self.bos:
                hyp.append(word_idx)
            k = self.prevKs[j][k]
        return hyp[::-1]

    def print_sentences(self, decoder_vocab):
        sentences = [[] for _ in range(self.size)]
        for t in range(len(self.nextYs)):
            ys = self.nextYs[t]
            for i in range(self.size):
                word_idx = int(ys[i].item())
                word = decoder_vocab._index_to_token['targets'][word_idx]
                sentences[i].append(word)
        for sent in sentences:
            print(' '.join(sent))
        print('\n')


def beam_search(decoder, encoder_outputs, encoder_outputs_mask, beam_size=BEAM_SIZE, debug=False):
    """ performs beam search. returns hypotheses with scores."""
    batch_size = encoder_outputs.size(0)
    encoder_hidden_dim = encoder_outputs.size(2)
    assert encoder_hidden_dim == decoder._decoder_hidden_dim

    trg_h_t, trg_c_t = decoder._initalize_hidden_context_states(
        encoder_outputs, encoder_outputs_mask)

    max_trg_length = decoder._max_decoding_steps

    # Expand tensors for each beam.
    dec_states = [
        Variable(trg_h_t.data.repeat(beam_size, 1)),  # [bs * beam_size, h]
        Variable(trg_c_t.data.repeat(beam_size, 1))
    ]

    beam = [
        Beam(beam_size, decoder.vocab, cuda=True)
        for k in range(batch_size)
    ]

    batch_idx = list(range(batch_size))
    remaining_sents = batch_size

    for i in range(max_trg_length):
        input_indices = torch.stack(
            [b.get_current_state() for b in beam if not b.done]
        ).contiguous().view(-1)  # [beam_size*bs]

        # need to repeat inside since update_active()
        encoder_outputs_beam = encoder_outputs.repeat(beam_size, 1, 1)
        encoder_outputs_mask_beam = encoder_outputs_mask.repeat(beam_size, 1, 1)
        decoder_input = decoder._prepare_decode_step_input(
            input_indices=input_indices,
            decoder_hidden_state=dec_states[0],
            encoder_outputs=encoder_outputs_beam,
            encoder_outputs_mask=encoder_outputs_mask_beam,
        )

        logits, dec_states = decoder._decoder_step(
            decoder_input,
            dec_states[0],
            dec_states[1],
        )

        transition_probs = F.softmax(logits, dim=1)
        # be careful if you want to change this - the orientation doesn't
        # work if you switch dims in view() and remove transpose()
        word_lk = transition_probs.view(
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

        def update_active_encoder_out(t):
            return t.index_select(0, active_idx)
        encoder_outputs = update_active_encoder_out(encoder_outputs)
        encoder_outputs_mask = update_active_encoder_out(encoder_outputs_mask)

        remaining_sents = len(active)

    # package final best hypotheses
    allHyp, allScores = [], []
    for b in range(batch_size):
        scores, sort_idx = beam[b].sort_best()
        allScores.append(scores[0])
        best_hyp = beam[b].get_hyp(sort_idx[0])
        allHyp.append(best_hyp)

    if debug:
        for b in beam:
            b.print_sentences(decoder.vocab)

    return allHyp, allScores


def compute_bleu(hyps, scores, targets):
    # masking is handled by hardcoded check for zeros.
    relevant_targets = targets['words'].cpu().numpy()[1:]
    bleu_score = bleu_scoring.get_bleu(hyps, relevant_targets)
    return bleu_score


def generate_and_compute_bleu(decoder, encoder_outputs, encoder_outputs_mask,
                              targets):
    hyps, scores = beam_search(decoder, encoder_outputs, encoder_outputs_mask)
    bleu_score = compute_bleu(hyps, scores, targets)

    return bleu_score
