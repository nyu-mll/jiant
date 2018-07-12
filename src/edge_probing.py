# Implementation of edge probing module.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from .tasks import EdgeProbingTask
from . import modules

from allennlp.modules.span_extractors import \
        EndpointSpanExtractor

from typing import Dict


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
    except instead of whole-sentence pooling we'll use span1 and span2 indices.

    TODO: consider alternate span-pooling operators: SegRNN for each span, or
    3rd-order Tensor interaction term between spans.
    '''

    def __init__(self, task, d_inp: int, task_params):
        super(EdgeClassifierModule, self).__init__()
        # Set config options needed for forward pass.
        self.loss_type = task_params['cls_loss_fn']
        self.span_enum_mode = task_params['span_enum_mode']

        proj_dim = task_params['d_hid']
        # Separate projection for span1, span2
        self.proj1 = nn.Linear(d_inp, proj_dim)
        self.proj2 = nn.Linear(d_inp, proj_dim)

        # Span extractor, shared for both span1 and span2.
        # Shouldn't actually have any parameters with the default config.
        self.span_extractor = EndpointSpanExtractor(proj_dim, combination="x,y")

        # Classifier gets summed projections of span1, span2
        clf_input_dim = self.span_extractor.get_output_dim()  # 2 * proj_dim
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
        se_proj1 = self.proj1(sent_embs)
        se_proj2 = self.proj2(sent_embs)

        # Span extraction.
        #  span_mask = (batch['labels'] != -1)  # [batch_size, num_targets] bool
        span_mask = (batch['span1s'][:,:,0] != -1)  # [batch_size, num_targets] bool
        out['mask'] = span_mask
        total_num_targets = span_mask.sum()
        out['n_targets'] = total_num_targets
        out['n_exs'] = total_num_targets  # used by trainer.py

        _kw = dict(sequence_mask=sent_mask.long(),
                   span_indices_mask=span_mask.long())
        def _extract_spans(embs, spans):
            return self.span_extractor.forward(embs, spans, **_kw)
        # span1_emb and span2_emb are [batch_size, num_targets, clf_input_dim]
        span1_emb = _extract_spans(se_proj1, batch['span1s'])
        span2_emb = _extract_spans(se_proj2, batch['span2s'])

        # [batch_size, num_targets, n_classes]
        logits = self.classifier(span1_emb + span2_emb)
        out['logits'] = logits

        # Compute loss if requested.
        if 'labels' in batch:
            # Flatten to [total_num_targets, ...] first.
            out['loss'] = self.compute_loss(logits[span_mask],
                                            batch['labels'][span_mask],
                                            task)

        if predict:
            # return argmax predictions
            _, out['preds'] = logits.max(dim=1)

        return out


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
