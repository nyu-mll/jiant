""" Helper functions to evaluate a model on a dataset """
import os
import logging as log
import tqdm

import torch
from allennlp.data.iterators import BasicIterator
from utils import device_mapping
from tasks import STSBTask, JOCITask


def evaluate(model, tasks, batch_size, cuda_device, split="val"):
    '''Evaluate on a dataset'''
    model.eval()
    iterator = BasicIterator(batch_size)

    all_metrics = {"micro_accuracy": 0.0, "macro_accuracy": 0.0}
    all_preds = {}
    n_overall_examples = 0
    for task in tasks:
        n_examples = 0
        task_preds, task_idxs = [], []
        if split == "val":
            dataset = task.val_data
        elif split == 'train':
            dataset = task.train_data
        elif split == "test":
            dataset = task.test_data
        generator = iterator(dataset, num_epochs=1, shuffle=False, cuda_device=cuda_device)
        generator_tqdm = tqdm.tqdm(generator, total=iterator.get_num_batches(dataset), disable=True)
        for batch in generator_tqdm:
            tensor_batch = batch
            if 'idx' in tensor_batch:
                task_idxs += tensor_batch['idx'].data.tolist()
                tensor_batch.pop('idx', None)
            out = model.forward(task, tensor_batch)
            task_metrics = task.get_metrics()
            description = ', '.join(["%s_%s: %.2f" % (task.name, name, value) for name, value in
                                     task_metrics.items()]) + " ||"
            generator_tqdm.set_description(description)
            if 'labels' in batch:
                n_examples += batch['labels'].size()[0]
            else:
                assert 'targs' in batch
                n_examples += batch['targs']['words'].nelement()
            if isinstance(task, STSBTask) or isinstance(task, JOCITask):
                try:
                    preds, _ = out['logits'].max(dim=1)
                except BaseException:
                    pass
            else:
                _, preds = out['logits'].max(dim=1)
            task_preds += preds.data.tolist()

        task_metrics = task.get_metrics(reset=True)
        for name, value in task_metrics.items():
            all_metrics["%s_%s" % (task.name, name)] = value
        all_metrics["micro_accuracy"] += all_metrics[task.val_metric] * n_examples
        all_metrics["macro_accuracy"] += all_metrics[task.val_metric]
        n_overall_examples += n_examples
        if isinstance(task, STSBTask):
            task_preds = [min(max(0., pred * 5.), 5.) for pred in task_preds]
        all_preds[task.name] = (task_preds, task_idxs)

    all_metrics["macro_accuracy"] /= len(tasks)
    all_metrics["micro_accuracy"] /= n_overall_examples

    return all_metrics, all_preds


def write_preds(all_preds, pred_dir):
    ''' Write predictions to files in pred_dir

    We write special code to handle various GLUE tasks. '''
    for task, preds in all_preds.items():
        if task not in ['cola', 'sst', 'qqp', 'mrpc', 'sts-b', 'mnli', 'qnli', 'rte', 'wnli']:
            continue
        if isinstance(preds[1][0], list):
            preds = [[p for p in preds[0]], [p[0] for p in preds[1]]]
        idxs_and_preds = [(idx, pred) for pred, idx in zip(preds[0], preds[1])]
        idxs_and_preds.sort(key=lambda x: x[0])
        if task == 'mnli':
            pred_map = {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
            with open(os.path.join(pred_dir, "%s-m.tsv" % (task)), 'w') as pred_fh:
                pred_fh.write("index\tprediction\n")
                split_idx = 0
                for idx, pred in idxs_and_preds[:9796]:
                    pred = pred_map[pred]
                    pred_fh.write("%d\t%s\n" % (split_idx, pred))
                    split_idx += 1
            with open(os.path.join(pred_dir, "%s-mm.tsv" % (task)), 'w') as pred_fh:
                pred_fh.write("index\tprediction\n")
                split_idx = 0
                for idx, pred in idxs_and_preds[9796:9796 + 9847]:
                    pred = pred_map[pred]
                    pred_fh.write("%d\t%s\n" % (split_idx, pred))
                    split_idx += 1
            with open(os.path.join(pred_dir, "diagnostic.tsv"), 'w') as pred_fh:
                pred_fh.write("index\tprediction\n")
                split_idx = 0
                for idx, pred in idxs_and_preds[9796 + 9847:]:
                    pred = pred_map[pred]
                    pred_fh.write("%d\t%s\n" % (split_idx, pred))
                    split_idx += 1
        else:
            with open(os.path.join(pred_dir, "%s.tsv" % (task)), 'w') as pred_fh:
                pred_fh.write("index\tprediction\n")
                for idx, pred in idxs_and_preds:
                    if 'sts-b' in task:
                        pred_fh.write("%d\t%.3f\n" % (idx, pred))
                    elif 'rte' in task or 'qnli' in task:
                        pred = 'entailment' if pred else 'not_entailment'
                        pred_fh.write('%d\t%s\n' % (idx, pred))
                    else:
                        pred_fh.write("%d\t%d\n" % (idx, pred))

    return


def write_results(results, results_file, run_name):
    ''' Aggregate results by appending results to results_file '''
    with open(results_file, 'a') as results_fh:
        all_metrics_str = ', '.join(['%s: %.3f' % (metric, score) for
                                     metric, score in results.items()])
        results_fh.write("%s\t%s\n" % (run_name, all_metrics_str))
    log.info(all_metrics_str)


def load_model_state(model, state_path, gpu_id, skip_task_models=False):
    ''' Helper function to load a model state

    Parameters
    ----------
    model: The model object to populate with loaded parameters.
    state_path: The path to a model_state checkpoint.
    gpu_id: The GPU to use. -1 for no GPU.
    skip_task_models: If set, load only the task-independent parameters.
    '''
    model_state = torch.load(state_path, map_location=device_mapping(gpu_id))
    if skip_task_models:
        keys_to_skip = [key for key in model_state if "_mdl" in key]
        for key in keys_to_skip:
            del model_state[key]

    model.load_state_dict(model_state, strict=False)
    log.info("Loaded model state from %s", state_path)
