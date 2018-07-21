# most of this code is adapted from
# https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/evaluate.py

from collections import Counter
import math
import numpy as np
import sys


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    (c, r) = stats[:2]
    try:
        log_bleu_terms = [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2]) if x > 0 and y > 0]
        log_bleu_prec = sum(log_bleu_terms) / 4. if log_bleu_terms else 0
        return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)
    except ValueError:
        print(sys.exc_info()[0], stats)
        return 0


def get_bleu(hypotheses, reference, debug=False):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    bleu_score = 100 * bleu(stats)
    if debug:
        print(bleu_score)
    return bleu_score
