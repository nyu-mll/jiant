# Implementation of edge probing module.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from .tasks import EdgeProbingTask
from . import modules

from allennlp.modules.span_extractors import \
        EndpointSpanExtractor, SelfAttentiveSpanExtractor

from typing import Dict, Iterable, List


def to_onehot(labels: torch.Tensor, n_classes: int):
    """ Convert integer-valued labels to one-hot targets. """
    binary_targets = torch.zeros([labels.shape[0], n_classes],
                                 dtype=torch.int64, device=labels.device)
    ridx = torch.arange(labels.shape[0], dtype=torch.int64)
    binary_targets[ridx, labels] = 1
    return binary_targets

class EdgeClassifierModule(nn.Module):
    ''' Build edge classifier components as a sub-module.

    Use same classifier code as build_single_sentence_module,
    except instead of whole-sentence pooling we'll use span1 and span2 indices
    to extract span representations, and use these as input to the classifier.

    This works in the current form, but with some provisos:
        - Expects both span1 and span2 to be set. TODO to support single-span
        tasks like tagging and constituency membership.
        - Only considers the explicit set of spans in inputs; does not consider
        all other spans as negatives. (So, this won't work for argument
        _identification_ yet.)
        - Spans are represented by endpoints, and pooled by projecting each
        side and adding the project span1 and span2 representations (this is
        equivalent to concat + linear layer).

    TODO: consider alternate span-pooling operators: max or mean-pooling,
    soft-head pooling, or SegRNN.

    TODO: add span-expansion to negatives, one of the following modes:
        - all-spans (either span1 or span2), treating not-seen as negative
        - all-tokens (assuming span1 and span2 are length-1), e.g. for
        dependency parsing
        - batch-negative (pairwise among spans seen in batch, where not-seen
        are negative)
    '''
    def _make_span_extractor(self):
        if self.span_pooling == "attn":
            return SelfAttentiveSpanExtractor(self.proj_dim)
        else:
            return EndpointSpanExtractor(self.proj_dim,
                                         combination=self.span_pooling)

    def __init__(self, task, d_inp: int, task_params):
        super(EdgeClassifierModule, self).__init__()
        # Set config options needed for forward pass.
        self.loss_type = task_params['cls_loss_fn']
        self.span_pooling = task_params['cls_span_pooling']
        self.is_symmetric = task.is_symmetric

        self.proj_dim = task_params['d_hid']
        # Separate projection for span1, span2.
        # Use these to reduce dimensionality in case we're enumerating a lot of
        # spans - we want to do this *before* extracting spans for greatest
        # efficiency.
        self.proj1 = nn.Linear(d_inp, self.proj_dim)
        if self.is_symmetric:
            # Use None as dummy padding for readability,
            # so that we can index projs[1] and projs[2]
            self.projs = [None, self.proj1, self.proj1]
        else:
            # Separate params for span2
            self.proj2 = nn.Linear(d_inp, self.proj_dim)
            self.projs = [None, self.proj1, self.proj2]

        # Span extractor, shared for both span1 and span2.
        self.span_extractor1 = self._make_span_extractor()
        if self.is_symmetric:
            self.span_extractors = [None, self.span_extractor1, self.span_extractor1]
        else:
            self.span_extractor2 = self._make_span_extractor()
            self.span_extractors = [None, self.span_extractor1, self.span_extractor2]

        # Classifier gets concatenated projections of span1, span2
        clf_input_dim = (self.span_extractors[1].get_output_dim()
                         + self.span_extractors[2].get_output_dim())
        self.classifier = modules.Classifier.from_params(clf_input_dim,
                                                         task.n_classes,
                                                         task_params)

    def forward(self, batch: Dict,
                sent_embs: torch.Tensor,
                sent_mask: torch.Tensor,
                task: EdgeProbingTask,
                predict: bool) -> Dict:
        """ Run forward pass.

        Expects batch to have the following entries:
            'batch1' : [batch_size, max_len, ??]
            'labels' : [batch_size, num_targets] of label indices
            'span1s' : [batch_size, num_targets, 2] of spans
            'span2s' : [batch_size, num_targets, 2] of spans

        'labels', 'span1s', and 'span2s' are padded with -1 along second
        (num_targets) dimension.

        Args:
            batch: dict(str -> Tensor) with entries described above.
            sent_embs: [batch_size, max_len, repr_dim] Tensor
            sent_mask: [batch_size, max_len, 1] Tensor of {0,1}
            task: EdgeProbingTask
            predict: whether or not to generate predictions

        Returns:
            out: dict(str -> Tensor)
        """
        out = {}

        batch_size = sent_embs.shape[0]
        out['n_inputs'] = batch_size

        # Apply projection layers for each span.
        se_proj1 = self.projs[1](sent_embs)
        se_proj2 = self.projs[2](sent_embs)

        # Span extraction.
        #  span_mask = (batch['labels'] != -1)  # [batch_size, num_targets] bool
        span_mask = (batch['span1s'][:,:,0] != -1)  # [batch_size, num_targets] bool
        out['mask'] = span_mask
        total_num_targets = span_mask.sum()
        out['n_targets'] = total_num_targets
        out['n_exs'] = total_num_targets  # used by trainer.py

        _kw = dict(sequence_mask=sent_mask.long(),
                   span_indices_mask=span_mask.long())
        # span1_emb and span2_emb are [batch_size, num_targets, span_repr_dim]
        span1_emb = self.span_extractors[1](se_proj1, batch['span1s'], **_kw)
        span2_emb = self.span_extractors[2](se_proj2, batch['span2s'], **_kw)
        span_emb = torch.cat([span1_emb, span2_emb], dim=2)

        # [batch_size, num_targets, n_classes]
        logits = self.classifier(span_emb)
        out['logits'] = logits

        # Compute loss if requested.
        if 'labels' in batch:
            # Flatten to [total_num_targets, ...] first.
            out['loss'] = self.compute_loss(logits[span_mask],
                                            batch['labels'][span_mask],
                                            task)

        if predict:
            # Return preds as a list.
            preds = self.get_predictions(logits)
            out['preds'] = list(self.unbind_predictions(preds, span_mask))

        return out

    def unbind_predictions(self, preds: torch.Tensor,
                           masks: torch.Tensor) -> Iterable[np.ndarray]:
        """ Unpack preds to varying-length numpy arrays.

        Args:
            preds: [batch_size, num_targets, ...]
            masks: [batch_size, num_targets] boolean mask

        Yields:
            np.ndarray for each row of preds, selected by the corresponding row
            of span_mask.
        """
        preds = preds.detach().cpu()
        masks = masks.detach().cpu()
        for pred, mask in zip(torch.unbind(preds, dim=0),
                              torch.unbind(masks, dim=0)):
            yield pred[mask].numpy()  # only non-masked predictions


    def get_predictions(self, logits: torch.Tensor):
        if self.loss_type == 'softmax':
            # For softmax loss, return argmax predictions.
            # [batch_size, num_targets]
            return logits.max(dim=2)[1]  # argmax
        elif self.loss_type == 'sigmoid':
            # For sigmoid loss, return class probabilities.
            # [batch_size, num_targets, n_classes]
            return F.sigmoid(logits)
        else:
            raise ValueError("Unsupported loss type '%s' "
                             "for edge probing." % loss_type)

    def compute_loss(self, logits: torch.Tensor,
                     labels: torch.Tensor, task: EdgeProbingTask):
        """ Compute loss & eval metrics.

        Expect logits and labels to be already "selected" for good targets,
        i.e. this function does not do any masking internally.

        Args:
            logits: [total_num_targets, n_classes] Tensor of float scores
            labels: [total_num_targets] Tensor of int targets

        Returns:
            loss: scalar Tensor
        """
        # Accuracy scorer can handle multiclass natively.
        task.acc_scorer(logits, labels)

        # Matthews and F1 need binary targets, as does sigmoid loss.
        # [total_num_targets, n_classes] LongTensor
        binary_targets = to_onehot(labels, n_classes=logits.shape[1])

        # Matthews coefficient computed on {0,1} labels.
        task.mcc_scorer(logits.ge(0).long(), binary_targets)

        # F1Measure() expects [total_num_targets, n_classes, 2]
        # to compute binarized F1.
        binary_scores = torch.stack([-1*logits, logits], dim=2)
        task.f1_scorer(binary_scores, binary_targets)

        if self.loss_type == 'softmax':
            # softmax over each row.
            return F.cross_entropy(logits, flat_labels)
        elif self.loss_type == 'sigmoid':
            return F.binary_cross_entropy(F.sigmoid(logits),
                                          binary_targets.float())
        else:
            raise ValueError("Unsupported loss type '%s' "
                             "for edge probing." % loss_type)
