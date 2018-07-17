""" Helper functions to evaluate a model on a dataset """
import os
import logging as log

import pandas as pd

import torch
from allennlp.data.iterators import BasicIterator
from . import tasks
from . import preprocess
from .tasks import RegressionTask, STSBTask, JOCITask

from typing import List, Sequence, Tuple, Dict

def _coerce_list(preds) -> List:
    if isinstance(preds, torch.Tensor):
        return preds.data.tolist()
    else:
        return list(preds)

def parse_write_preds_arg(write_preds_arg: str) -> List[str]:
    if write_preds_arg == 0:
        return []
    elif write_preds_arg == 1:
        return ['test']
    else:
        return write_preds_arg.split(",")

def evaluate(model, tasks: Sequence[tasks.Task], batch_size: int,
             cuda_device: int, split="val") -> Tuple[Dict, pd.DataFrame]:
    '''Evaluate on a dataset'''
    FIELDS_TO_EXPORT = ['idx', 'sent1str']
    model.eval()
    iterator = BasicIterator(batch_size)

    all_metrics = {"micro_avg": 0.0, "macro_avg": 0.0}
    all_preds = {}
    n_examples_overall = 0
    for task in tasks:
        log.info("Evaluating on: %s", task.name)
        n_examples = 0
        task_preds = []  # accumulate DataFrames
        assert split in ["train", "val", "test"]
        dataset = getattr(task, "%s_data" % split)
        generator = iterator(dataset, num_epochs=1, shuffle=False, cuda_device=cuda_device)
        for batch in generator:

            out = model.forward(task, batch, predict=True)
            n_examples += out["n_exs"]

            # get predictions
            if 'preds' not in out:
                continue
            preds = _coerce_list(out['preds'])
            assert isinstance(preds, list), "Convert predictions to list!"
            cols = {"preds": preds}
            for field in FIELDS_TO_EXPORT:
                cols[field] = _coerce_list(batch[field])
            # Transpose data using Pandas
            df = pd.DataFrame(cols)
            task_preds.append(df)
        # task_preds will be a DataFrame with columns ['preds'] +
        # FIELDS_TO_EXPORT
        # for GLUE tasks, preds entries should be single scalars.

        # Combine task_preds from each batch to a single DataFrame.
        task_preds = pd.concat(task_preds, ignore_index=True)

        # Update metrics
        task_metrics = task.get_metrics(reset=True)
        for name, value in task_metrics.items():
            all_metrics["%s_%s" % (task.name, name)] = value
        all_metrics["micro_avg"] += all_metrics[task.val_metric] * n_examples
        all_metrics["macro_avg"] += all_metrics[task.val_metric]
        n_examples_overall += n_examples

        # Store predictions, sorting by index if given.
        if 'idxs' in task_preds.columns:
           task_preds.sort_values(by=['idxs'], inplace=True)

        all_preds[task.name] = task_preds

    all_metrics["micro_avg"] /= n_examples_overall
    all_metrics["macro_avg"] /= len(tasks)

    return all_metrics, all_preds

def write_preds(all_preds, pred_dir, split_name) -> None:
    for task_name, preds_df in all_preds.items():
        if task_name in preprocess.ALL_GLUE_TASKS + ['wmt']:
            write_glue_preds(task_name, preds_df, pred_dir, split_name)
            log.info("Task '%s': Wrote predictions to %s", task_name, pred_dir)
        else:
            log.warning("Task '%s' not supported by write_preds().",
                        task_name)
            continue
    log.info("Wrote all preds for split '%s' to %s", split_name, pred_dir)
    return


def write_glue_preds(task_name, preds_df, pred_dir, split_name):
    ''' Write predictions to separate files located in pred_dir.
    We write special code to handle various GLUE tasks.

    TODO: clean this up, remove special cases & hard-coded dataset sizes.

    Args:
        - all_preds (Dict[str:list]): dictionary mapping task names to predictions.
            Assumes that predictions are sorted (if necessary).
            For tasks with sentence predictions, we assume they've been mapped back to strings.
    '''

    def _write_preds_to_file(preds, indices, sent1_strs,
                             pred_file, pred_map=None, write_type=int):
        ''' Write preds to pred_file '''
        with open(pred_file, 'w') as pred_fh:
            pred_fh.write("index\tprediction\tsentence_1\n")
            for idx, pred in enumerate(preds):
                index = indices[idx] if len(indices) > 0 else -1
                sent1_str = sent1_strs[idx] if len(sent1_strs) > 0 else ""
                if pred_map is not None or write_type == str:
                    pred = pred_map[pred]
                    pred_fh.write("%d\t%s\t%s\n" % (index, pred, sent1_str))
                elif write_type == float:
                    pred_fh.write("%d\t%.3f\t%s\n" % (index, pred, sent1_str))
                elif write_type == int:
                    pred_fh.write("%d\t%d\t%s\n" % (index, pred, sent1_str))

    if len(preds_df) == 0:  # catch empty lists
        log.warning("Task '%s': predictions are empty!", task_name)
        return

    log.info("Wrote predictions for task: %s", task_name)
    preds = preds_df['preds']
    indices = preds_df['idx']
    sent1_strs = preds_df['sent1str']
    default_output_filename = os.path.join(pred_dir,
                                           "%s__%s.tsv" % (task_name,
                                                           split_name))

    if task_name == 'mnli' and split_name == 'test':  # 9796 + 9847 + 1104 = 20747
        assert len(preds) == 20747, "Missing predictions for MNLI!"
        log.info("There are %d examples in MNLI, 20747 were expected")
        pred_map = {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
        _write_preds_to_file(preds[:9796], indices, sent1_strs,
                             os.path.join(pred_dir, "%s-m.tsv" % task_name),
                             pred_map)
        _write_preds_to_file(preds[9796:19643], indices, sent1_strs,
                             os.path.join(pred_dir, "%s-mm.tsv" % task_name),
                             pred_map=pred_map)
        _write_preds_to_file(preds[19643:], indices, sent1_strs,
                             os.path.join(pred_dir, "diagnostic.tsv"), pred_map)

    elif task_name in ['rte', 'qnli']:
        pred_map = {0: 'not_entailment', 1: 'entailment'}
        _write_preds_to_file(preds, indices, sent1_strs,
                             default_output_filename, pred_map)
    elif task_name in ['sts-b']:
        preds = [min(max(0., pred * 5.), 5.) for pred in preds]
        _write_preds_to_file(preds, indices, sent1_strs,
                             default_output_filename, write_type=float)
    elif task_name in ['wmt']:
        # convert each prediction to a single string if we find a list of tokens
        if isinstance(preds[0], list):
            assert isinstance(preds[0][0], str)
            preds = [' '.join(pred) for pred in preds]
        _write_preds_to_file(preds, indices, sent1_strs,
                             default_output_filename, write_type=str)
    else:
        _write_preds_to_file(preds, indices, sent1_strs,
                             default_output_filename)


def write_results(results, results_file, run_name):
    ''' Aggregate results by appending results to results_file '''
    with open(results_file, 'a') as results_fh:
        all_metrics_str = ', '.join(['%s: %.3f' % (metric, score) for
                                     metric, score in results.items()])
        results_fh.write("%s\t%s\n" % (run_name, all_metrics_str))
    log.info(all_metrics_str)
