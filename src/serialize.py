#!/bin/bash

# Serialization and deserialization helpers.
# Write arbitrary pickle-able Python objects to a record file, with one object 
# per line as a base64-encoded pickle.

import _pickle as pkl
import base64

def _serialize(examples, fd, flush_every):
    for i, example in enumerate(examples):
        blob = pkl.dumps(example)
        encoded = base64.b64encode(blob)
        fd.write(encoded)
        fd.write(b"\n")
        if (i + 1) % flush_every == 0 and hasattr(fd, 'flush'):
            fd.flush()

def write_records(examples, filename, flush_every=10000):
	"""Streaming read records from file.

	Args:
      examples: iterable(object), iterable of examples to write
      filename: path to file to write
      flush_every: (int), flush to disk after this many examples consumed
    """
    with open(filename, 'wb') as fd:
        _serialize(examples, fd, flush_every)

def _deserialize(fd):
    for line in fd:
        blob = base64.b64decode(line)
        example = pkl.loads(blob)
        yield example

def read_records(filename):
	"""Streaming read records from file.

	Args:
      filename: path to file of b64-encoded pickles, one per line

	Yields:
      deserialized Python objects
    """
	with open(filename, 'rb') as fd:
		return _deserialize(fd)

