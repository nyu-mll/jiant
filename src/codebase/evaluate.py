"""
The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ python -m allennlp.run evaluate --help
    usage: run [command] evaluate [-h] --archive_file ARCHIVE_FILE
                                --evaluation_data_file EVALUATION_DATA_FILE
                                [--cuda_device CUDA_DEVICE]

    Evaluate the specified model + dataset

    optional arguments:
    -h, --help            show this help message and exit
    --archive_file ARCHIVE_FILE
                            path to an archived trained model
    --evaluation_data_file EVALUATION_DATA_FILE
                            path to the file containing the evaluation data
    --cuda_device CUDA_DEVICE
                            id of GPU to use (if any)
"""
from typing import Dict, Any
import argparse
import logging

import tqdm

from allennlp.data import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn.util import arrays_to_variables

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def evaluate(model: Model,
             tasks,
             iterator: DataIterator,
             cuda_device: int,
             split="val") -> Dict[str, Any]:
    model.eval()

    all_metrics = {}
    for task in tasks:
        if split == "val":
            dataset = task.val_data
        elif split == "test":
            dataset = task.test_data
        generator = iterator(dataset, num_epochs=1)
        logger.info("Iterating over dataset")
        generator_tqdm = tqdm.tqdm(
                generator, total=iterator.get_num_batches(dataset))
        for batch in generator_tqdm:
            tensor_batch = arrays_to_variables(
                    batch, cuda_device, for_training=False)
            model.forward(task.pred_layer, task.pair_input,
                          scorer=task.scorer, **tensor_batch)
            task_metrics = task.get_metrics()
            description = ', '.join(["%s_%s: %.2f" % 
                (task.name, name, value) for name, value in 
                task_metrics.items()]) + " ||"
            generator_tqdm.set_description(description)

        task_metrics = task.get_metrics()
        for name, value in task_metrics.items():
            all_metrics["%s_%s" % (task.name, name)] = value

    return all_metrics

