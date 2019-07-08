import logging
from typing import Dict, Union

import numpy
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides
from torch.autograd import Variable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NumericField(Field[numpy.ndarray]):
    """
    A ``LabelField`` is a categorical label of some kind, where the labels are either strings of
    text or 0-indexed integers (if you wish to skip indexing by passing skip_indexing=True).
    If the labels need indexing, we will use a :class:`Vocabulary` to convert the string labels
    into integers.

    This field will get converted into an integer index representing the class label.

    Parameters
    ----------
    label : ``Union[str, int]``
    label_namespace : ``str``, optional (default="labels")
        The namespace to use for converting label strings into integers.  We map label strings to
        integers for you (e.g., "entailment" and "contradiction" get converted to 0, 1, ...),
        and this namespace tells the ``Vocabulary`` object which mapping from strings to integers
        to use (so "entailment" as a label doesn't get the same integer id as "entailment" as a
        word).  If you have multiple different label fields in your data, you should make sure you
        use different namespaces for each one, always using the suffix "labels" (e.g.,
        "passage_labels" and "question_labels").
    skip_indexing : ``bool``, optional (default=False)
        If your labels are 0-indexed integers, you can pass in this flag, and we'll skip the indexing
        step.  If this is ``False`` and your labels are not strings, this throws a ``ConfigurationError``.
    """

    def __init__(self, label: Union[float, int], label_namespace: str = "labels") -> None:
        self.label = label
        self._label_namespace = label_namespace
        self._label_id = numpy.array(label, dtype=numpy.float32)
        if not (self._label_namespace.endswith("labels") or self._label_namespace.endswith("tags")):
            logger.warning(
                "Your label namespace was '%s'. We recommend you use a namespace "
                "ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by "
                "default to your vocabulary.  See documentation for "
                "`non_padded_namespaces` parameter in Vocabulary.",
                self._label_namespace,
            )

    # idk what this is for
    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._label_id is None:
            counter[self._label_namespace][self.label] += 1  # type: ignore

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
        return {}

    def as_array(
        self, padding_lengths: Dict[str, int]
    ) -> numpy.ndarray:  # pylint: disable=unused-argument
        return numpy.asarray([self._label_id])

    @overrides
    def as_tensor(
        self, padding_lengths: Dict[str, int], cuda_device: int = -1, for_training: bool = True
    ) -> torch.Tensor:  # pylint: disable=unused-argument
        label_id = self._label_id.tolist()
        tensor = Variable(torch.FloatTensor([label_id]), volatile=not for_training)
        return tensor if cuda_device == -1 else tensor.cuda(cuda_device)

    @overrides
    def empty_field(self):
        return NumericField(0, self._label_namespace)
