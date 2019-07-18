# Implementation of edge probing module.

from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor

from jiant.tasks.edge_probing import EdgeProbingTask
from jiant.modules.simple_modules import Classifier


class EdgeClassifierModule(nn.Module):
    """ Build edge classifier components as a sub-module.

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
    """

    def _make_span_extractor(self):
        if self.span_pooling == "attn":
            return SelfAttentiveSpanExtractor(self.proj_dim)
        else:
            return EndpointSpanExtractor(self.proj_dim, combination=self.span_pooling)

    def _make_cnn_layer(self, d_inp):
        """Make a CNN layer as a projection of local context.

        CNN maps [batch_size, max_len, d_inp]
        to [batch_size, max_len, proj_dim] with no change in length.
        """
        k = 1 + 2 * self.cnn_context
        padding = self.cnn_context
        return nn.Conv1d(
            d_inp,
            self.proj_dim,
            kernel_size=k,
            stride=1,
            padding=padding,
            dilation=1,
            groups=1,
            bias=True,
        )

    def __init__(self, task, d_inp: int, task_params):
        super(EdgeClassifierModule, self).__init__()
        # Set config options needed for forward pass.
        self.loss_type = task_params["cls_loss_fn"]
        self.span_pooling = task_params["cls_span_pooling"]
        self.cnn_context = task_params["edgeprobe_cnn_context"]
        self.is_symmetric = task_params["edgeprobe_symmetric"]
        self.single_sided = task.single_sided

        self.proj_dim = task_params["d_hid"]
        # Separate projection for span1, span2.
        # Convolution allows using local context outside the span, with
        # cnn_context = 0 behaving as a per-word linear layer.
        # Use these to reduce dimensionality in case we're enumerating a lot of
        # spans - we want to do this *before* extracting spans for greatest
        # efficiency.
        self.proj1 = self._make_cnn_layer(d_inp)
        if self.is_symmetric or self.single_sided:
            # Use None as dummy padding for readability,
            # so that we can index projs[1] and projs[2]
            self.projs = [None, self.proj1, self.proj1]
        else:
            # Separate params for span2
            self.proj2 = self._make_cnn_layer(d_inp)
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
        self.classifier = Classifier.from_params(clf_input_dim, task.n_classes, task_params)

    def forward(
        self,
        batch: Dict,
        word_embs_in_context: torch.Tensor,
        sent_mask: torch.Tensor,
        task: EdgeProbingTask,
        predict: bool,
    ) -> Dict:
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
            word_embs_in_context: [batch_size, max_len, repr_dim] Tensor
            sent_mask: [batch_size, max_len, 1] Tensor of {0,1}
            task: EdgeProbingTask
            predict: whether or not to generate predictions

        Returns:
            out: dict(str -> Tensor)
        """
        out = {}

        batch_size = word_embs_in_context.shape[0]
        out["n_inputs"] = batch_size

        # Apply projection CNN layer for each span.
        word_embs_in_context_t = word_embs_in_context.transpose(1, 2)  # needed for CNN layer
        se_proj1 = self.projs[1](word_embs_in_context_t).transpose(2, 1).contiguous()
        if not self.single_sided:
            se_proj2 = self.projs[2](word_embs_in_context_t).transpose(2, 1).contiguous()

        # Span extraction.
        # [batch_size, num_targets] bool
        span_mask = batch["span1s"][:, :, 0] != -1
        out["mask"] = span_mask
        total_num_targets = span_mask.sum()
        out["n_targets"] = total_num_targets
        out["n_exs"] = total_num_targets  # used by trainer.py

        _kw = dict(sequence_mask=sent_mask.long(), span_indices_mask=span_mask.long())
        # span1_emb and span2_emb are [batch_size, num_targets, span_repr_dim]
        span1_emb = self.span_extractors[1](se_proj1, batch["span1s"], **_kw)
        if not self.single_sided:
            span2_emb = self.span_extractors[2](se_proj2, batch["span2s"], **_kw)
            span_emb = torch.cat([span1_emb, span2_emb], dim=2)
        else:
            span_emb = span1_emb

        # [batch_size, num_targets, n_classes]
        logits = self.classifier(span_emb)
        out["logits"] = logits

        # Compute loss if requested.
        if "labels" in batch:
            # Labels is [batch_size, num_targets, n_classes],
            # with k-hot encoding provided by AllenNLP's MultiLabelField.
            # Flatten to [total_num_targets, ...] first.
            out["loss"] = self.compute_loss(logits[span_mask], batch["labels"][span_mask], task)

        if predict:
            # Return preds as a list.
            preds = self.get_predictions(logits)
            out["preds"] = list(self.unbind_predictions(preds, span_mask))

        return out

    def unbind_predictions(self, preds: torch.Tensor, masks: torch.Tensor) -> Iterable[np.ndarray]:
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
        for pred, mask in zip(torch.unbind(preds, dim=0), torch.unbind(masks, dim=0)):
            yield pred[mask].numpy()  # only non-masked predictions

    def get_predictions(self, logits: torch.Tensor):
        """Return class probabilities, same shape as logits.

        Args:
            logits: [batch_size, num_targets, n_classes]

        Returns:
            probs: [batch_size, num_targets, n_classes]
        """
        if self.loss_type == "sigmoid":
            return torch.sigmoid(logits)
        else:
            raise ValueError("Unsupported loss type '%s' " "for edge probing." % self.loss_type)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, task: EdgeProbingTask):
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
        binary_scores = torch.stack([-1 * logits, logits], dim=2)
        task.f1_scorer(binary_scores, labels)

        if self.loss_type == "sigmoid":
            return F.binary_cross_entropy(torch.sigmoid(logits), labels.float())
        else:
            raise ValueError("Unsupported loss type '%s' " "for edge probing." % self.loss_type)
