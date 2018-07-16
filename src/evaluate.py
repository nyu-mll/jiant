""" Helper functions to evaluate a model on a dataset """
import os
import logging as log

import torch
from allennlp.data.iterators import BasicIterator
from .tasks import RegressionTask, STSBTask, JOCITask

def evaluate(model, tasks, batch_size, cuda_device, split="val"):
    '''Evaluate on a dataset'''
    model.eval()
    iterator = BasicIterator(batch_size)

    all_metrics, all_preds = {"micro_avg": 0.0, "macro_avg": 0.0}, {}
    n_examples_overall = 0
    for task in tasks:
        log.info("Evaluating on: %s", task.name)
        n_examples = 0
        task_preds, task_idxs, sen1_strs = [], [], []
        assert split in ["train", "val", "test"]
        dataset = getattr(task, "%s_data" % split)
        generator = iterator(dataset, num_epochs=1, shuffle=False, cuda_device=cuda_device)
        for batch in generator:

            out = model.forward(task, batch, predict=True)
            n_examples += out["n_exs"]

            # get predictions
            if 'preds' in out:
                preds = out['preds']
                if isinstance(preds, torch.Tensor):
                    preds = preds.data.tolist()
                assert isinstance(preds, list), "Convert predictions to list!"
                task_preds += preds
                if 'idx' in batch:
                    task_idxs += batch['idx'].data.tolist()
                if 'sent1str' in batch:
                    sen1_strs += batch['sent1str']
        # task_preds should be a list of length = # examples.
        # for GLUE tasks, entries should be single scalars.

        # Update metrics
        task_metrics = task.get_metrics(reset=True)
        for name, value in task_metrics.items():
            all_metrics["%s_%s" % (task.name, name)] = value
        all_metrics["micro_avg"] += all_metrics[task.val_metric] * n_examples
        all_metrics["macro_avg"] += all_metrics[task.val_metric]
        n_examples_overall += n_examples

        # Store predictions, possibly sorting them if
        if task_idxs:
            assert len(task_idxs) == len(task_preds), (
                "Number of indices != number of predictions!")
            idxs_preds_strs = [(idx, pred, sent_str) for idx, pred, sent_str in zip(task_idxs, task_preds, sen1_strs)]
            idxs_preds_strs.sort(key=lambda x: x[0])
            task_idxs, task_preds, sen1_strs = zip(*idxs_preds_strs)
        all_preds[task.name] = [task_preds, task_idxs, sen1_strs]

    all_metrics["micro_avg"] /= n_examples_overall
    all_metrics["macro_avg"] /= len(tasks)

    return all_metrics, all_preds

def write_preds(all_preds, pred_dir, split):
    ''' Write predictions to separate files located in pred_dir.
    We write special code to handle various GLUE tasks.

    TODO: clean this up, remove special cases & hard-coded dataset sizes.

    Args:
        - all_preds (Dict[str:list]): dictionary mapping task names to predictions.
            Assumes that predictions are sorted (if necessary).
            For tasks with sentence predictions, we assume they've been mapped back to strings.
    '''

    def write_preds_to_file(preds, indices, sent1_strs, pred_file, pred_map=None, write_type=int):
        ''' Write preds to pred_file '''
        with open(pred_file, 'w') as pred_fh:
            pred_fh.write("index\tprediction\n")
            for idx, pred in enumerate(preds):
                index = indices[idx] if indices else -1 # if there is no index it will write -1 for index column
                sent1_str = sent1_strs[idx] if sent1_strs else ""
                if pred_map is not None or write_type == str:
                    pred = pred_map[pred]
                    pred_fh.write("%d\t%s\t%s\n" % (index, pred, sent1_str))

                elif write_type == float:
                    pred_fh.write("%d\t%.3f\t%s\n" % (index, pred, sent1_str))
                elif write_type == int:
                    pred_fh.write("%d\t%d\t%s\n" % (index, pred, sent1_str))

    for task, preds in all_preds.items():
        if not preds:  # catch empty lists
            continue

        log.info("Wrote predictions for task: %s", task)
        indices = preds[1]
        sent1_strs = preds[2]
        preds = preds[0]


        if task == 'mnli' and split == 'test':  # 9796 + 9847 + 1104 = 20747
            assert len(preds) == 20747, "Missing predictions for MNLI!"
            log.info("There are %d examples in MNLI, 20747 were expected")
            pred_map = {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
            write_preds_to_file(preds[:9796], indices, sent1_strs, os.path.join(pred_dir, "%s-m.tsv" % task), pred_map)
            write_preds_to_file(preds[9796:19643], indices, sent1_strs, os.path.join(pred_dir, "%s-mm.tsv" % task),
                                pred_map=pred_map)
            write_preds_to_file(preds[19643:], indices, sent1_strs, os.path.join(pred_dir, "diagnostic.tsv"), pred_map)

        elif task in ['rte', 'qnli']:
            pred_map = {0: 'not_entailment', 1: 'entailment'}
            write_preds_to_file(preds, indices, sent1_strs, os.path.join(pred_dir, "%s__%s.tsv" % (task, split)), pred_map)
        elif task in ['sts-b']:
            preds = [min(max(0., pred * 5.), 5.) for pred in preds]
            write_preds_to_file(preds, indices, sent1_strs, os.path.join(pred_dir, "%s__%s.tsv" % (task, split)), write_type=float)
        elif task in ['wmt']:
            # convert each prediction to a single string if we find a list of tokens
            if isinstance(preds[0], list):
                assert isinstance(preds[0][0], str)
                preds = [' '.join(pred) for pred in preds]
            write_preds_to_file(preds, indices, sent1_strs, os.path.join(pred_dir, "%s__%s.tsv" % (task, split)), write_type=str)
        else:
            write_preds_to_file(preds, indices, sent1_strs, os.path.join(pred_dir, "%s__%s.tsv" % (task, split)))
    log.info("Wrote predictions to %s", pred_dir)
    return


def write_results(results, results_file, run_name):
    ''' Aggregate results by appending results to results_file '''
    with open(results_file, 'a') as results_fh:
        all_metrics_str = ', '.join(['%s: %.3f' % (metric, score) for
                                     metric, score in results.items()])
        results_fh.write("%s\t%s\n" % (run_name, all_metrics_str))
    log.info(all_metrics_str)
