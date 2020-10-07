import numpy as np
from typing import NamedTuple, Mapping, Dict
from enum import Enum
from dataclasses import dataclass

import torch
import torch.utils.data.dataloader as dataloader

from jiant.utils.python.datastructures import ExtendedDataClassMixin, combine_dicts


@dataclass
class FeaturizationSpec:
    """Tokenization-related metadata.

    Attributes:
        max_seq_length (int): The maximum total input sequence length after tokenization.
        cls_token_at_end (bool): True if class token is located at end, False if at beginning.
        pad_on_left (bool): True if padding is applied to left side, False if on the right side.
        cls_token_segment_id (int): int used to represent class token segment.
        pad_token_segment_id (int): int used to represent padding segments.
        pad_token_id (int): int used to represent the pad token in the tokenized representation.
        pad_token_mask_id (int): int used as a mask representing padding.
        sequence_a_segment_id (int): int used to represent sequence a segment.
        sequence_b_segment_id (int): int used to represent sequence b segment.
        sep_token_extra (bool): True for extra separators (e.g., "</s></s>"), false otherwise.

    """

    max_seq_length: int
    cls_token_at_end: bool
    pad_on_left: bool
    cls_token_segment_id: int
    pad_token_segment_id: int
    pad_token_id: int
    pad_token_mask_id: int
    sequence_a_segment_id: int
    sequence_b_segment_id: int
    sep_token_extra: bool


class BatchMixin(ExtendedDataClassMixin):
    def to(self, device, non_blocking=False, copy=False):
        # noinspection PyArgumentList
        return self.__class__(
            **{
                k: self._val_to_device(v=v, device=device, non_blocking=non_blocking, copy=copy,)
                for k, v in self.to_dict().items()
            }
        )

    @classmethod
    def _val_to_device(cls, v, device, non_blocking=False, copy=False):
        if isinstance(v, torch.Tensor):
            return v.to(device=device, non_blocking=non_blocking, copy=copy)
        else:
            return v

    def __len__(self):
        return len(getattr(self, self.get_fields()[0]))


class BaseExample(ExtendedDataClassMixin):
    def tokenize(self, tokenizer):
        raise NotImplementedError


class BaseTokenizedExample(ExtendedDataClassMixin):
    def featurize(self, tokenizer, feat_spec: FeaturizationSpec):
        raise NotImplementedError


class BaseDataRow(ExtendedDataClassMixin):
    pass


class BaseBatch(BatchMixin, ExtendedDataClassMixin):
    @classmethod
    def from_data_rows(cls, data_row_ls: list):
        raise NotImplementedError


def data_row_collate_fn(batch):
    assert isinstance(batch[0], BaseDataRow)


class TaskTypes(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2
    SPAN_COMPARISON_CLASSIFICATION = 3
    MULTIPLE_CHOICE = 4
    SPAN_CHOICE_PROB_TASK = 5
    SQUAD_STYLE_QA = 6
    TAGGING = 7
    MASKED_LANGUAGE_MODELING = 8
    EMBEDDING = 9
    MULTI_LABEL_SPAN_CLASSIFICATION = 10
    SPAN_PREDICTION = 11
    UNDEFINED = -1


class BatchTuple(NamedTuple):
    batch: BatchMixin
    metadata: dict

    def to(self, device, non_blocking=False, copy=False):
        return BatchTuple(
            batch=self.batch.to(device=device, non_blocking=non_blocking, copy=copy,),
            metadata=self.metadata,
        )


def metadata_collate_fn(metadata: list):
    return {k: [d[k] for d in metadata] for k in metadata[0].keys()}


def flat_collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, (np.ndarray, int, float, str)):
        return dataloader.default_collate(batch)
    elif isinstance(elem, (list, dict, set)):
        # Don't do anything to list of lists
        return batch
    else:
        raise TypeError(type(elem))


class Task:
    Example = NotImplemented
    TokenizedExample = NotImplemented
    DataRow = NotImplemented
    Batch = NotImplemented

    TASK_TYPE = NotImplemented

    def __init__(self, name: str, path_dict: dict):
        self.name = name
        self.path_dict = path_dict

    @property
    def train_path(self):
        return self.path_dict["train"]

    @property
    def val_path(self):
        return self.path_dict["val"]

    @property
    def test_path(self):
        return self.path_dict["test"]

    @classmethod
    def collate_fn(cls, batch):
        # cls.collate_fn
        elem = batch[0]
        if isinstance(elem, Mapping):  # dict
            assert set(elem.keys()) == {"data_row", "metadata"}
            data_rows = [x["data_row"] for x in batch]
            metadata = [x["metadata"] for x in batch]
            collated_data_rows = {
                key: flat_collate_fn([getattr(d, key) for d in data_rows])
                for key in data_rows[0].to_dict()
            }
            collated_metadata = metadata_collate_fn(metadata)
            combined = combine_dicts([collated_data_rows, collated_metadata])
            batch_dict = {}
            for field, field_type in cls.Batch.get_annotations().items():
                batch_dict[field] = combined.pop(field)
                if field_type == torch.FloatTensor:
                    # Ensure that floats stay as float32
                    batch_dict[field] = batch_dict[field].float()
            out_batch = cls.Batch(**batch_dict)
            remainder = combined
            return out_batch, remainder
        else:
            raise TypeError(f"Unknown type for collate_fn {type(elem)}")


class SuperGlueMixin:
    """Mixin for :class:`jiant.tasks.core.Task`s in the `SuperGLUE<https://super.gluebenchmark.com/>`_
    benchmark.
    """

    @classmethod
    def super_glue_format_preds(cls, pred_dict: Dict):
        """Tasks in the SuperGLUE benchmark must implement this method to return
        predictions in the expected SuperGLUE format. This function is expect to
        return a List[Dict] with each Dict value in the format expected by the
        SuperGLUE benchmark submission grader.

        Args:
            pred_dict (Dict): Dictionary mapping "preds" to List of label ids
            and
            "guids" to List of example ids

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()


class GlueMixin:
    """Mixin for :class:`jiant.tasks.core.Task`s in the `GLUE<https://gluebenchmark.com/>`_
    benchmark.
    """

    @classmethod
    def get_glue_preds(cls, pred_dict: Dict):
        """Returns a tuple of (index, prediction) for GLUE benchmark submission.

        Args:
            pred_dict (Dict): Dictionary mapping "preds" to label ids and "guids"
            to example ids

        Returns:
            (tuple): tuple containing:
                indexes (List[int]): example ids
                predictions (List[str]): prediction labels (in original task
                format)
        """
        indexes = []
        predictions = []
        for pred, guid in zip(list(pred_dict["preds"]), list(pred_dict["guids"])):
            indexes.append(int(guid.split("-")[1]))
            predictions.append(str(cls.LABELS[pred]).lower())
        return (indexes, predictions)
