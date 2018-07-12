# Implementation of edge probing module.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from .tasks import EdgeProbingTask
from . import modules

from typing import Dict

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
        # Build separate projection blocks, will apply once to sentence embeddings
        # for each type (start1, end1, start2, end2).
        self.proj_s1 = nn.Linear(d_inp, proj_dim)
        self.proj_e1 = nn.Linear(d_inp, proj_dim)
        self.proj_s2 = nn.Linear(d_inp, proj_dim)
        self.proj_e2 = nn.Linear(d_inp, proj_dim)
        # Classifier gets summed projections of (start1, end1, start2, end2)
        self.classifier = modules.Classifier.from_params(proj_dim, 
                                                         task.n_classes,
                                                         task_params)

    def forward(self, batch: Dict, 
                sent_embs: torch.Tensor,
                sent_mask: torch.Tensor,
                task: EdgeProbingTask, 
                predict: bool) -> Dict:
        out = {}

        # embed the sentence
        # batch['input1'] is [batch_size, max_len, dim???]
        # sent_embs is [batch_size, max_len, repr_dim]
        # sent_mask is [batch_size, max_len, 1], boolean mask
        #  sent_embs, sent_mask = sent_encoder(batch['input1'])

        # Apply projection layer to sent_embs.
        # Get four different versions, which we can combine to span
        # representations.
        se_proj_s1 = self.proj_s1(sent_embs)
        se_proj_e1 = self.proj_e1(sent_embs)
        se_proj_s2 = self.proj_s2(sent_embs)
        se_proj_e2 = self.proj_e2(sent_embs)

        # batch['labels'] is [batch_size, num_targets] array of ints
        # padded with -1 along last dimension.

        # batch['span1s'] and batch['span2s'] are [batch_size, num_targets, 2]
        # array of ints, padded with -1 along second dimension.

        # For now, just extract specific spans and use a sigmoid loss.
        # TODO to enumerate all spans and take the given targets as positives.
        batch_size = sent_embs.shape[0]
        max_targets = batch['span1s'].shape[1]
        mask = (batch['labels'] != -1)  # [batch_size, num_targets] bool
        total_num_targets = mask.sum()
        out['n_inputs'] = batch_size
        out['n_targets'] = total_num_targets
        out['n_exs'] = total_num_targets  # used by trainer.py
        # [batch_size, max_targets] of batch indices
        batch_idxs = (torch.arange(batch_size, dtype=torch.int64)
                      .repeat(max_targets, 1)
                      .transpose(0,1))
        # Get indices in preparation for a big gather.
        flat_batch_idxs = batch_idxs[mask]   # [total_num_targets]
        flat_span1s = batch['span1s'][mask]  # [total_num_targets, 2]
        flat_span2s = batch['span2s'][mask]  # [total_num_targets, 2]

        # RaSoR-style pooling: first and last token for each span.
        # Results are [total_num_targets, repr_dim]
        # TODO: implement more efficiently by learning separate projection
        # layers from repr_dim -> classifier_hid_dim for starts, ends
        # and applying this to the sentence reprs separately before combining
        # into span representations.
        #  s1 = sent_embs[flat_batch_idxs, flat_span1s[:,0]]
        #  e1 = sent_embs[flat_batch_idxs, flat_span1s[:,1]]
        #  s2 = sent_embs[flat_batch_idxs, flat_span2s[:,0]]
        #  e2 = sent_embs[flat_batch_idxs, flat_span2s[:,1]]
        #  # [total_num_targets, 4*repr_dim]
        #  span_vecs = torch.cat([s1, e1, s2, e2], dim=1)
        s1 = se_proj_s1[flat_batch_idxs, flat_span1s[:, 0]]
        e1 = se_proj_e2[flat_batch_idxs, flat_span1s[:, 1]]
        s2 = se_proj_s1[flat_batch_idxs, flat_span2s[:, 0]]
        e2 = se_proj_e2[flat_batch_idxs, flat_span2s[:, 1]]

        # mimic first layer of a classifier by summing projected embs
        span_vecs = s1 + e1 + s2 + e2  # [total_num_targets, proj_dim]
        logits = self.classifier(span_vecs)  # [total_num_targets, n_classes]
        out['logits'] = logits

        # Compute loss if requested.
        # TODO(iftenney): replace with sigmoid loss.
        if 'labels' in batch:
            flat_labels = batch['labels'][mask]  # [total_num_targets]
            task.acc_scorer(logits, flat_labels)

            # binary cross-entropy against one-hot rows.
            binary_targets = torch.zeros_like(logits, dtype=torch.float32)
            ridx = torch.arange(flat_labels.shape[0], dtype=torch.int64)
            # [total_num_targets, n_classes]
            binary_targets[ridx, flat_labels] = 1

            # needed for AllenNLP F1Measure()
            # shape [total_num_targets, n_classes, 2], with float scores
            binary_preds = torch.stack([-1*logits, logits], dim=2)
            task.f1_scorer(binary_preds, binary_targets)
            # Matthews coefficient computed on {0,1} labels.
            task.mcc_scorer(logits.ge(0).long(), binary_targets.long())

            if self.loss_type == 'softmax':
                # softmax over each row.
                out['loss'] = F.cross_entropy(logits, flat_labels)
            elif self.loss_type == 'sigmoid':
                out['loss'] = F.binary_cross_entropy(F.sigmoid(logits),
                                                     binary_targets)
            else:
                raise ValueError("Unsupported loss type '%s' "
                                 "for edge probing." % loss_type)

        if predict:
            # return argmax predictions
            _, out['preds'] = logits.max(dim=1)

        return out


