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
import pdb
import logging
import tqdm

from allennlp.data.iterators import DataIterator
from allennlp.models.model import Model

from tasks import STSBenchmarkTask, STS14Task

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def evaluate(model, tasks, iterator, cuda_device, split="val"):
    '''Evaluate on a dataset'''
    model.eval()

    all_metrics = {"micro_accuracy": 0.0, "macro_accuracy": 0.0}
    all_preds = {}
    n_overall_examples = 0
    for task in tasks:
        n_examples = 0
        task_preds = []
        if split == "val":
            dataset = task.val_data
        elif split == 'train':
            dataset = task.train_data
        elif split == "test":
            dataset = task.test_data
        generator = iterator(dataset, num_epochs=1, shuffle=False, cuda_device=cuda_device)
        generator_tqdm = tqdm.tqdm(generator, total=iterator.get_num_batches(dataset), disable=True)
        for batch in generator_tqdm:
            #tensor_batch = arrays_to_variables(batch, cuda_device, for_training=False)
            tensor_batch = batch
            out = model.forward(task, **tensor_batch)
            task_metrics = task.get_metrics()
            description = ', '.join(["%s_%s: %.2f" % (task.name, name, value) for name, value in
                                     task_metrics.items()]) + " ||"
            generator_tqdm.set_description(description)
            n_examples += batch['label'].size()[0]
            _, preds = out['logits'].max(dim=1)
            task_preds += preds.data.tolist()

        task_metrics = task.get_metrics(reset=True)
        for name, value in task_metrics.items():
            all_metrics["%s_%s" % (task.name, name)] = value
        all_metrics["micro_accuracy"] += all_metrics["%s_accuracy" % task.name] * n_examples
        all_metrics["macro_accuracy"] += all_metrics["%s_accuracy" % task.name]
        n_overall_examples += n_examples
        if isinstance(task, (STSBenchmarkTask, STS14Task)):
            task_preds = [pred * 5. for pred in task_preds]
        all_preds[task.name] = task_preds

    all_metrics["macro_accuracy"] /= len(tasks)
    all_metrics["micro_accuracy"] /= n_overall_examples

    return all_metrics, all_preds
