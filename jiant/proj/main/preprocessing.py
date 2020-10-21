import numpy as np
import torch

import jiant.shared.caching as shared_caching
import jiant.utils.torch_utils as torch_utils
from jiant.tasks.core import FeaturizationSpec, TaskTypes
from jiant.utils.display import maybe_tqdm, maybe_trange


class MaxValidLengthRecorder:
    def __init__(self, max_seq_length):
        self.max_valid_length = 0
        self.max_seq_length = max_seq_length
        self.range_idx = np.arange(max_seq_length)

    def __call__(self, datum):
        if "input_mask" not in datum["data_row"].get_fields():
            raise RuntimeError("Smart truncate not supported")
        indexer = datum["data_row"].input_mask.reshape(-1, self.max_seq_length).max(-2)
        valid_length = self.range_idx[indexer.astype(bool)].max() + 1
        self.max_valid_length = max(self.max_valid_length, valid_length)


def smart_truncate(dataset: torch_utils.ListDataset, max_seq_length: int, verbose: bool = False):
    """Truncate data to the length of the longest example in the dataset.

    Args:
        dataset (torch_utils.ListDataset): ListDataset to truncate if possible.
        max_seq_length (int): The maximum total input sequence length.
        verbose (bool): If True, display progress bar tracking truncation progress.

    Returns:
        Tuple[torch_utils.ListDataset, int]: truncated dataset, and length of the longest sequence.

    """
    if "input_mask" not in dataset.data[0]["data_row"].get_fields():
        raise RuntimeError("Smart truncate not supported")
    valid_length_ls = []
    range_idx = np.arange(max_seq_length)
    for datum in dataset.data:
        # TODO: document why reshape and max happen here (for cola this isn't necessary).
        #       (issue #1185)
        indexer = datum["data_row"].input_mask.reshape(-1, max_seq_length).max(-2)
        valid_length_ls.append(range_idx[indexer.astype(bool)].max() + 1)
    max_valid_length = max(valid_length_ls)

    if max_valid_length == max_seq_length:
        return dataset, max_seq_length

    new_datum_ls = []
    for datum in maybe_tqdm(dataset.data, desc="Smart truncate data", verbose=verbose):
        new_datum_ls.append(
            smart_truncate_datum(
                datum=datum, max_seq_length=max_seq_length, max_valid_length=max_valid_length,
            )
        )
    new_dataset = torch_utils.ListDataset(new_datum_ls)
    return new_dataset, max_valid_length


def smart_truncate_cache(
    cache: shared_caching.ChunkedFilesDataCache,
    max_seq_length: int,
    max_valid_length: int,
    verbose: bool = False,
):
    for chunk_i in maybe_trange(cache.num_chunks, desc="Smart truncate chunks", verbose=verbose):
        chunk = torch.load(cache.get_chunk_path(chunk_i))
        new_chunk = []
        for datum in maybe_tqdm(chunk, desc="Smart truncate chunk-datum", verbose=verbose):
            new_chunk.append(
                smart_truncate_datum(
                    datum=datum, max_seq_length=max_seq_length, max_valid_length=max_valid_length,
                )
            )
        torch.save(new_chunk, cache.get_chunk_path(chunk_i))


def smart_truncate_datum(datum, max_seq_length, max_valid_length):
    row_dict = datum["data_row"].to_dict()
    new_row_dict = row_dict.copy()
    for k, v in row_dict.items():
        if not isinstance(v, np.ndarray):
            continue
        if max_seq_length not in v.shape:
            continue
        if not v.shape.count(max_seq_length) == 1:
            raise RuntimeError("confusing dimensions")
        slice_ls = []
        for n in v.shape:
            if n == max_seq_length:
                slice_ls.append(slice(None, max_valid_length))
            else:
                slice_ls.append(slice(None))
        new_row_dict[k] = v[tuple(slice_ls)]
    return {
        "data_row": datum["data_row"].__class__(**new_row_dict),
        "metadata": datum["metadata"],
    }


def convert_examples_to_dataset(
    task, examples: list, tokenizer, feat_spec: FeaturizationSpec, phase: str, verbose=False
):
    """Create ListDataset containing DataRows and metadata.

    Args:
        task (Task): Task object
        examples (list[Example]): list of task Examples.
        tokenizer: TODO  (issue #1188)
        feat_spec (FeaturizationSpec): Tokenization-related metadata.
        phase (str): string identifying the data subset (e.g., train, val or test).
        verbose: If True, display progress bar.

    Returns:
        ListDataset containing DataRows and metadata.

    """
    data_rows = tokenize_and_featurize(
        task=task,
        examples=examples,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
        phase=phase,
        verbose=verbose,
    )
    metadata = {"example_id": list(range(len(data_rows)))}
    data = []
    for i, data_row in enumerate(data_rows):
        metadata_row = {k: v[i] for k, v in metadata.items()}
        data.append({"data_row": data_row, "metadata": metadata_row})
    return torch_utils.ListDataset(data)


def iter_chunk_convert_examples_to_dataset(
    task, examples: list, tokenizer, feat_spec: FeaturizationSpec, phase: str, verbose=False
):
    for i, data_row in enumerate(
        iter_chunk_tokenize_and_featurize(
            task=task,
            examples=examples,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            phase=phase,
            verbose=verbose,
        )
    ):
        metadata = {"example_id": i}
        yield {"data_row": data_row, "metadata": metadata}


def tokenize_and_featurize(
    task, examples: list, tokenizer, feat_spec: FeaturizationSpec, phase, verbose=False
):
    """Create list of DataRows containing tokenized and featurized examples.

    Args:
        task (Task): Task object
        examples (list[Example]): list of task Examples.
        tokenizer: TODO  (issue #1188)
        feat_spec (FeaturizationSpec): Tokenization-related metadata.
        phase (str): string identifying the data subset (e.g., train, val or test).
        verbose: If True, display progress bar.

    Returns:
        List DataRows containing tokenized and featurized examples.

    """
    # TODO: Better solution  (issue #1184)
    if task.TASK_TYPE == TaskTypes.SQUAD_STYLE_QA:
        data_rows = []
        for example in maybe_tqdm(examples, desc="Tokenizing", verbose=verbose):
            data_rows += example.to_feature_list(
                tokenizer=tokenizer,
                max_seq_length=feat_spec.max_seq_length,
                doc_stride=task.doc_stride,
                max_query_length=task.max_query_length,
                set_type=phase,
            )
    else:
        data_rows = [
            example.tokenize(tokenizer).featurize(tokenizer, feat_spec)
            for example in maybe_tqdm(examples, desc="Tokenizing", verbose=verbose)
        ]
    return data_rows


def iter_chunk_tokenize_and_featurize(
    task, examples: list, tokenizer, feat_spec: FeaturizationSpec, phase, verbose=False
):
    """Generator of DataRows containing tokenized and featurized examples.

    Args:
        task (Task): Task object
        examples (list[Example]): list of task Examples.
        tokenizer: TODO  (issue #1188)
        feat_spec (FeaturizationSpec): Tokenization-related metadata.
        phase (str): string identifying the data subset (e.g., train, val or test).
        verbose: If True, display progress bar.

    Yields:
        DataRow containing tokenized and featurized examples.

    """
    for example in maybe_tqdm(examples, desc="Tokenizing", verbose=verbose):
        # TODO: Better solution  (issue #1184)
        if task.TASK_TYPE == TaskTypes.SQUAD_STYLE_QA:
            yield from example.to_feature_list(
                tokenizer=tokenizer,
                max_seq_length=feat_spec.max_seq_length,
                doc_stride=task.doc_stride,
                max_query_length=task.max_query_length,
                set_type=phase,
            )
        else:
            yield example.tokenize(tokenizer).featurize(tokenizer, feat_spec)
