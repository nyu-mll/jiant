# Serialization and deserialization helpers.
# Write arbitrary pickle-able Python objects to a record file, with one object
# per line as a base64-encoded pickle.

import _pickle as pkl
import base64
from zlib import crc32


def _serialize(examples, fd, flush_every):
    for i, example in enumerate(examples):
        blob = pkl.dumps(example)
        encoded = base64.b64encode(blob)
        fd.write(encoded)
        fd.write(b"\n")
        if (i + 1) % flush_every == 0 and hasattr(fd, "flush"):
            fd.flush()


def write_records(examples, filename, flush_every=10000):
    """Streaming read records from file.

    Args:
      examples: iterable(object), iterable of examples to write
      filename: path to file to write
      flush_every: (int), flush to disk after this many examples consumed
    """
    with open(filename, "wb") as fd:
        _serialize(examples, fd, flush_every)


class RepeatableIterator(object):
    """Repeatable iterator class."""

    def __init__(self, iter_fn):
        """Create a repeatable iterator.

        Args:
          iter_fn: callable with no arguments, creates an iterator
        """
        self._iter_fn = iter_fn
        self._counter = 0

    def get_counter(self):
        return self._counter

    def __iter__(self):
        self._counter += 1
        return self._iter_fn().__iter__()


def bytes_to_float(b):
    """ Maps a byte string to a float in [0, 1].

    Verified to be uniform, at least over text strings and zero byte strings of varying lengths.
    """
    return float(crc32(b) & 0xFFFFFFFF) / 2 ** 32


def read_records(filename, repeatable=False, fraction=None):
    """Streaming read records from file.

    Args:
      filename: path to file of b64-encoded pickles, one per line
      repeatable: if true, returns a RepeatableIterator that can read the file
        multiple times.
      fraction: if set to a float between 0 and 1, load only the specified percentage
        of examples. Hashing is used to ensure that the same examples are loaded each
        epoch.

    Returns:
      iterable, possible repeatable, yielding deserialized Python objects
    """

    def _iter_fn():
        with open(filename, "rb") as fd:
            for line in fd:
                blob = base64.b64decode(line)
                if fraction and fraction < 1:
                    hash_float = bytes_to_float(blob)
                    if hash_float > fraction:
                        continue
                example = pkl.loads(blob)
                yield example

    return RepeatableIterator(_iter_fn) if repeatable else _iter_fn()
