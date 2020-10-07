import math
import numpy as np
import os
from typing import Generator, Union, Sequence

import torch
import torch.utils.data.dataset


class Chunker:
    def __init__(self, length, num_chunks, chunk_size):
        self.length = length
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size

    def get_slices(self):
        indices = list(range(0, self.length, self.chunk_size)) + [self.length]
        return [slice(start, end) for start, end in zip(indices[:-1], indices[1:])]

    def get_chunks(self, data):
        assert len(data) == self.length
        chunked_data = [data[data_slice] for data_slice in self.get_slices()]
        assert len(chunked_data) == self.num_chunks
        return chunked_data

    def lookup_chunk_and_index(self, i):
        if isinstance(i, int):
            return i // self.chunk_size, i % self.chunk_size
        elif isinstance(i, np.ndarray):
            i = i.astype(int)
            return (i / self.chunk_size).astype(int), (i % self.chunk_size).astype(int)
        elif isinstance(i, torch.Tensor):
            return self.lookup_chunk_and_index(i.numpy())
        else:
            raise TypeError(type(i))

    def lookup_index(self, chunk_i, i):
        if isinstance(i, (int, np.ndarray, torch.Tensor)):
            return chunk_i * self.chunk_size + i
        else:
            raise TypeError(type(i))

    @classmethod
    def from_chunk_size(cls, length, chunk_size):
        num_chunks = math.ceil(length / chunk_size)
        return cls(length=length, num_chunks=num_chunks, chunk_size=chunk_size)


def convert_to_chunks(data, chunk_size: int):
    """Divide data into chunks.

    Args:
        data (List): data to divide into chunks.
        chunk_size (int): number of data elements to store per chunk.

    Returns:
        List of data chunks.
    """
    chunker = Chunker.from_chunk_size(len(data), chunk_size=chunk_size)
    chunked_data = chunker.get_chunks(data)
    return chunked_data


def chunk_and_save(data: list, chunk_size: int, data_args: dict, output_dir: str):
    """Divide data into chunks and save it to disk, also saves metadata describing chunking to disk.

    Args:
        data (List): List of DataRows and metadata.
        chunk_size (int): number of data elements to store per chunk.
        data_args (Dict): RunConfiguration represented as a dictionary.
        output_dir: phase-specific dir in the output dir specified in the RunConfiguration.

    """
    os.makedirs(output_dir, exist_ok=True)
    chunked_data = convert_to_chunks(data=data, chunk_size=chunk_size)
    for i, chunk in enumerate(chunked_data):
        torch.save(chunk, os.path.join(output_dir, f"data_{i:05d}.chunk"))
    data_args = data_args.copy()
    data_args["num_chunks"] = len(chunked_data)
    data_args["length"] = len(data)
    torch.save(data_args, os.path.join(output_dir, "data_args.p"))


def iter_chunk_and_save(
    data: Generator, chunk_size: int, data_args: dict, output_dir: str, recorder_callback=None
):
    os.makedirs(output_dir, exist_ok=True)
    chunk_i = 0
    length = 0
    current_chunk = []
    for datum in data:
        if recorder_callback is not None:
            recorder_callback(datum)
        length += 1
        current_chunk.append(datum)
        if len(current_chunk) == chunk_size:
            torch.save(current_chunk, os.path.join(output_dir, f"data_{chunk_i:05d}.chunk"))
            chunk_i += 1
            current_chunk = []
    if current_chunk:
        torch.save(current_chunk, os.path.join(output_dir, f"data_{chunk_i:05d}.chunk"))
        chunk_i += 1
    data_args = data_args.copy()
    data_args["num_chunks"] = chunk_i
    data_args["length"] = length
    torch.save(data_args, os.path.join(output_dir, "data_args.p"))


def compare_tensor_tuples(tup1, tup2):
    if len(tup1) != len(tup2):
        return False
    for col1, col2 in zip(tup1, tup2):
        if not torch.equal(col1, col2):
            return False
    return True


def compare_dataset_with_metadata(d1, d2):
    if not compare_tensor_tuples(d1.dataset.tensors, d2.dataset.tensors):
        return False
    if not d1.metadata == d2.metadata:
        return False
    return True


class DataCache:
    # We're going to liberally use pickling/torch.save/load.
    # There is no expectation that caches should be backward compatible.

    def get_all(self):
        raise NotImplementedError()

    def iter_all(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class InMemoryDataCache(DataCache):
    def __init__(self, data):
        self.data = data

    def get_all(self):
        return self.data

    def iter_all(self):
        for elem in self.data:
            yield elem

    def __len__(self):
        return len(self.data)


class ChunkedFilesDataCache(DataCache):
    def __init__(self, cache_fol_path):
        self.cache_fol_path = cache_fol_path

        self.data_args = torch.load(os.path.join(cache_fol_path, "data_args.p"))
        self.num_chunks = self.data_args["num_chunks"]
        self.length = self.data_args["length"]
        self.chunk_size = self.data_args["chunk_size"]
        self.chunker = Chunker.from_chunk_size(length=self.length, chunk_size=self.chunk_size)

    def get_iterable_dataset(
        self,
        buffer_size=None,
        shuffle=False,
        subset_num: Union[None, int] = None,
        explicit_subset: Union[None, Sequence] = None,
        verbose=False,
    ):
        return ChunkedFilesIterableDataset(
            buffer_size=buffer_size,
            shuffle=shuffle,
            subset_num=subset_num,
            explicit_subset=explicit_subset,
            chunked_file_data_cache=self,
            verbose=verbose,
        )

    def load_chunk(self, i):
        return torch.load(self.get_chunk_path(i))

    def get_chunk_path(self, i):
        return os.path.join(self.cache_fol_path, f"data_{i:05d}.chunk")

    def load_from_indices(self, indices, verbose=False):
        chunk_arr, chunk_sub_index_arr = self.chunker.lookup_chunk_and_index(indices)
        reverse_index = np.arange(len(indices)).astype(int)
        result = [None] * len(indices)
        for chunk_i in sorted(list(set(chunk_arr))):
            selector = chunk_arr == chunk_i
            chunk = self.load_chunk(chunk_i)
            selected_chunk_sub_index_arr = chunk_sub_index_arr[selector]
            selected_reverse_index = reverse_index[selector]
            if verbose:
                print(f"Loading {len(selected_chunk_sub_index_arr)} indices from chunk {chunk_i}")
            for i, j in zip(selected_chunk_sub_index_arr, selected_reverse_index):
                result[j] = chunk[i]
            del chunk
        return result

    def get_all(self):
        data = []
        for i in range(self.num_chunks):
            data += list(self.load_chunk(i))
        return data

    def iter_all(self):
        for i in range(self.num_chunks):
            chunk = self.load_chunk(i)
            for elem in chunk:
                yield elem

    def __len__(self):
        return self.length


class ChunkedFilesIterableDataset(torch.utils.data.dataset.IterableDataset):
    def __init__(
        self,
        buffer_size,
        shuffle,
        chunked_file_data_cache: ChunkedFilesDataCache,
        subset_num: Union[int, None] = None,
        explicit_subset: Union[Sequence, None] = None,
        verbose=False,
    ):
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.subset_num = subset_num
        self.chunked_file_data_cache = chunked_file_data_cache
        self.explicit_subset = explicit_subset
        self.verbose = verbose

        if self.explicit_subset is not None:
            assert self.subset_num is None
            self.length = len(self.explicit_subset)
        else:
            self.length = self.chunked_file_data_cache.length
            if self.subset_num:
                self.length = min(self.subset_num, self.length)

        if self.buffer_size is None:
            self.buffer_size = self.length

    def __iter__(self):
        seen = 0
        buffer_chunked_indices = self.get_buffer_chunked_indices()
        for buffer_chunked_index in buffer_chunked_indices:
            if self.verbose:
                print(
                    f"Loading buffer {seen} - {seen + len(buffer_chunked_index)}"
                    f" out of {len(self)}"
                )
            buffer = self.chunked_file_data_cache.load_from_indices(
                buffer_chunked_index, verbose=self.verbose
            )
            for elem in buffer:
                yield elem
            seen += len(buffer_chunked_index)

    def get_buffer_chunked_indices(self):
        if self.explicit_subset is not None:
            indices = np.array(self.explicit_subset).astype(int)
        else:
            indices = np.arange(self.length).astype(int)
        if self.shuffle:
            np.random.shuffle(indices)
        if self.subset_num:
            indices = indices[: self.subset_num]
        buffer_chunked_indices = convert_to_chunks(indices, chunk_size=self.buffer_size)
        return buffer_chunked_indices

    def __len__(self):
        return self.length
