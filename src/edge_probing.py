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

SPAN_CONSTANT = 10 # move this

class EdgeClassifierModule(nn.Module):
    ''' Build edge classifier components as a sub-module.

    Use same classifier code as build_single_sentence_module,
    except instead of whole-sentence pooling we'll use span1 and span2 indices
    to extract span representations, and use these as input to the classifier.

    This works in the current form, but with some provisos:
        - Only considers the explicit set of spans in inputs; does not consider
        all other spans as negatives. (So, this won't work for argument
        _identification_ yet.)

    TODO: consider alternate span-pooling operators: max or mean-pooling,
    or SegRNN.

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
        self.single_sided = task.single_sided
        self.max_seq_len = task.max_seq_len
        self.detect_spans = task.detect_spans

        self.proj_dim = task_params['d_hid']
        # Separate projection for span1, span2.
        # Use these to reduce dimensionality in case we're enumerating a lot of
        # spans - we want to do this *before* extracting spans for greatest
        # efficiency.
        self.proj1 = nn.Linear(d_inp, self.proj_dim)
        if self.is_symmetric or self.single_sided:
            # Use None as dummy padding for readability,
            # so that we can index projs[1] and projs[2]
            self.projs = [None, self.proj1, self.proj1]
        else:
            # Separate params for span2
            self.proj2 = nn.Linear(d_inp, self.proj_dim)
            self.projs = [None, self.proj1, self.proj2]

        # Span extractor, shared for both span1 and span2.
        self.span_extractor1 = self._make_span_extractor()
        if self.is_symmetric or self.single_sided:
            self.span_extractors = [None, self.span_extractor1, self.span_extractor1]
        else:
            self.span_extractor2 = self._make_span_extractor()
            self.span_extractors = [None, self.span_extractor1, self.span_extractor2]

        # Classifier gets concatenated projections of span1, span2
        clf_input_dim = self.span_extractors[1].get_output_dim()
        if not self.single_sided:
            clf_input_dim += self.span_extractors[2].get_output_dim()
        self.classifier = modules.Classifier.from_params(clf_input_dim,
                                                         task.n_classes,
                                                         task_params)
        if self.detect_spans:
            print (f"max seq len: {self.max_seq_len}")
            index_array = np.array([[[i, i + j] for j in range(SPAN_CONSTANT)] for i in range(self.max_seq_len)])
            self.index_array = torch.from_numpy(index_array)

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
        seq_len = sent_embs.shape[1]
        out['n_inputs'] = batch_size

        # Apply projection layers for each span.
        se_proj1 = self.projs[1](sent_embs)
        if not self.single_sided:
            se_proj2 = self.projs[2](sent_embs)

        # Span extraction.
        span_mask = (batch['span1s'][:,:,0] != -1)  # [batch_size, num_targets] bool
        out['mask'] = span_mask
        if self.detect_spans:
            total_num_targets = batch_size * seq_len * SPAN_CONSTANT
        else:
            total_num_targets = span_mask.sum()
        out['n_targets'] = total_num_targets
        out['n_exs'] = total_num_targets  # used by trainer.py
        if self.detect_spans:
            # [batch_size * seq_len * seq_len * 2]
            index_array_mask = torch.from_numpy(np.array([[int(span[1] < seq_len) for span in span_start_at]
                                                          for span_start_at in self.index_array[:seq_len, :]])).cuda()
            _kw = dict(sequence_mask=sent_mask.long())
            candidate_spans = self.index_array[:seq_len,:].repeat(batch_size, 1, 1, 1).view(batch_size,
                                                                                            seq_len * SPAN_CONSTANT, -1)

            flat_index_array_mask = index_array_mask.view(seq_len * SPAN_CONSTANT, 1).byte().cpu()
            masked_spans = torch.masked_select(candidate_spans, flat_index_array_mask).view(batch_size, -1, 2)
            num_spans = masked_spans.shape[1]
            assert self.single_sided, "Span detection currently only implemented for single_sided"
            span_emb = self.span_extractors[1](se_proj1,
                                               masked_spans.cuda(),
                                               **_kw) # [batch_size * seq_len * SPAN_CONSTANT * emb_size]
            # this part is in numpy; could possibly be moved to preprocessing
            if 'labels' in batch:
                label_size = batch['labels'].shape[-1]
                np_labels = np.zeros([batch_size, seq_len * SPAN_CONSTANT, label_size], dtype=np.uint8)
                for batch_num in range(batch_size):
                    for span_idx, span in enumerate(batch['span1s'][batch_num]):
                        if span_mask[batch_num][span_idx] != 0:
                            np_labels[batch_num][span[0] * SPAN_CONSTANT + (span[1] - span[0])] = batch['labels'][batch_num][span_idx]
                labels = torch.masked_select(torch.from_numpy(np_labels),
                                             flat_index_array_mask).view(batch_size, num_spans, -1).cuda()
                # create a new span mask that uses _all_ of it
                span_mask = torch.ones([batch_size, num_spans], dtype=torch.uint8)
        else:
            _kw = dict(sequence_mask=sent_mask.long(),
                       span_indices_mask=span_mask.long())
            print (sent_mask.size(), span_mask.size(), batch['span1s'].size())
            # span1_emb and span2_emb are [batch_size, num_targets, span_repr_dim]
            span1_emb = self.span_extractors[1](se_proj1, batch['span1s'], **_kw)
            if not self.single_sided:
                span2_emb = self.span_extractors[2](se_proj2, batch['span2s'], **_kw)
                span_emb = torch.cat([span1_emb, span2_emb], dim=2)
            else:
                span_emb = span1_emb
            if 'labels' in batch:
                labels = batch['labels']

        logits = self.classifier(span_emb)
        out['logits'] = logits

        # Compute loss if requested.
        if 'labels' in batch:
            # Labels is [batch_size, num_targets, n_classes],
            # with k-hot encoding provided by AllenNLP's MultiLabelField.
            # Flatten to [total_num_targets, ...] first.
            out['loss'] = self.compute_loss(logits[span_mask],
                                            labels[span_mask],
                                            task)
            # print (out['loss'])

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
        """Return class probabilities, same shape as logits.

        Args:
            logits: [batch_size, num_targets, n_classes]

        Returns:
            probs: [batch_size, num_targets, n_classes]
        """
        if self.loss_type == 'softmax':
            raise NotImplementedError("Softmax loss not fully supported.")
            return F.softmax(logits, dim=2)
        elif self.loss_type == 'sigmoid':
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
            labels: [total_num_targets, n_classes] Tensor of sparse binary targets

        Returns:
            loss: scalar Tensor
        """
        binary_preds = logits.ge(0).long()  # {0,1}

        # Matthews coefficient and accuracy computed on {0,1} labels.
        task.mcc_scorer(binary_preds, labels.long())
        task.acc_scorer(binary_preds, labels.long())

        # F1Measure() expects [total_num_targets, n_classes, 2]
        # to compute binarized F1.
        binary_scores = torch.stack([-1*logits, logits], dim=2)
        task.f1_scorer(binary_scores, labels)

        if self.loss_type == 'softmax':
            raise NotImplementedError("Softmax loss not fully supported.")
            # Expect exactly one target, convert to indices.
            assert labels.shape[1] == 1  # expect a single target
            return F.cross_entropy(logits, labels)
        elif self.loss_type == 'sigmoid':
            return F.binary_cross_entropy(F.sigmoid(logits),
                                          labels.float())
        else:
            raise ValueError("Unsupported loss type '%s' "
                             "for edge probing." % self.loss_type)
