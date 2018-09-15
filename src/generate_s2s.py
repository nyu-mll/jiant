import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from torch.autograd import Variable
import torch.nn.functional as F

from . import bleu_scoring
import numpy as np


def _get_word(decoder_vocab, word_idx):
    return decoder_vocab._index_to_token['targets'][word_idx]


def greedy_search(decoder, encoder_outputs, encoder_outputs_mask, debug=False):
    """ performs beam search. returns hypotheses with scores."""
    batch_size = encoder_outputs.size(0)
    encoder_hidden_dim = encoder_outputs.size(2)
    assert encoder_hidden_dim == decoder._decoder_hidden_dim

    trg_h_t, trg_c_t = decoder._initalize_hidden_context_states(
        encoder_outputs, encoder_outputs_mask)

    max_trg_length = decoder._max_decoding_steps

    # Expand tensors for each beam.
    dec_states = (trg_h_t, trg_c_t)
    gen_indices = encoder_outputs.new_zeros(batch_size, max_trg_length + 1).fill_(
        decoder.vocab.get_token_index(START_SYMBOL, "targets"))

    for i in range(1, max_trg_length):
        decoder_input = decoder._prepare_decode_step_input(
            input_indices=gen_indices[:, i - 1],
            decoder_hidden_state=dec_states[0],
            encoder_outputs=encoder_outputs,
            encoder_outputs_mask=encoder_outputs_mask,
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
            batch_size,
            -1
        )
        scores, gen_indices[:, i] = word_lk.max(1)  # TODO calculate scores

    def _print_sentence(indices):
        sent = [_get_word(decoder.vocab, word_idx.item()) for word_idx in indices[1:]]
        print(' '.join(sent))

    if debug:
        for i in range(gen_indices.size(0)):
            _print_sentence(gen_indices[i, :])

    return gen_indices.cpu().numpy(), scores


def write_translation_preds(hyps, relevant_targets, preds_file_path, decoder_vocab):
    with open(preds_file_path, "a") as f:
        for i in range(len(hyps)):
            # form hyp sentence
            hyp_sentence = []
            for word_idx in hyps[i, 1:]:
                hyp_sentence.append(
                    _get_word(decoder_vocab, word_idx)
                )
                if word_idx == decoder_vocab.get_token_index(END_SYMBOL, "targets"):
                    break
            hyp_sentence = ' '.join(hyp_sentence)

            target = relevant_targets[i]
            target_sentence = ' '.join([_get_word(decoder_vocab, i) for i in target])
            f.write('{}\t{}\n'.format(hyp_sentence, target_sentence))


def compute_bleu(hyps, scores, relevant_targets):
    raise NotImplementedError("BLEU score not yet working!")


def generate_and_compute_bleu(decoder, encoder_outputs, encoder_outputs_mask,
                              targets, preds_file_path):
    hyps, scores = greedy_search(decoder, encoder_outputs, encoder_outputs_mask)

    # important - preprocess targets for relevant targets
    # masking is handled by hardcoded check for zeros.
    relevant_targets = [[int(wordidx.item()) for i, wordidx in enumerate(
        target) if wordidx != 0 and i > 0] for target in targets]

    targets_unk_ratio = [len([i for i in target if i == decoder._unk_index]
                             ) / len(target) for target in targets]
    unk_ratio_macroavg = np.mean(targets_unk_ratio)

    bleu_score = compute_bleu(hyps, scores, relevant_targets)
    write_translation_preds(hyps, relevant_targets, preds_file_path, decoder_vocab=decoder.vocab)

    return bleu_score, unk_ratio_macroavg
