""" Helper functions to evaluate a model on a dataset """
import os
import ipdb as pdb
import logging as log

import torch
from allennlp.data.iterators import BasicIterator
from tasks import RegressionTask, STSBTask, JOCITask


def evaluate(model, tasks, batch_size, cuda_device, split="val"):
    '''Evaluate on a dataset'''
    model.eval()
    iterator = BasicIterator(batch_size)

    all_metrics, all_preds = {"micro_avg": 0.0, "macro_avg": 0.0}, {}
    n_overall_examples = 0
    for task in tasks:
        n_examples = 0
        task_preds, task_idxs = [], []
        assert split in ["train", "val", "test"]
        dataset = getattr(task, "%s_data" % split)
        generator = iterator(dataset, num_epochs=1, shuffle=False, cuda_device=cuda_device)
        for batch in generator:
            if 'idx' in batch: # for sorting examples
                task_idxs += batch['idx'].data.tolist()
                batch.pop('idx', None)
            out = model.forward(task, batch)

            # count number of examples; currently not working for sequence tasks
            if 'labels' in batch:
                n_examples += batch['labels'].size()[0]
            else:
                assert 'targs' in batch
                n_examples += batch['targs']['words'].nelement()

            # get predictions
            if isinstance(task, (RegressionTask, STSBTask, JOCITask)):
                preds, _ = out['logits'].max(dim=1)
            else:
                _, preds = out['logits'].max(dim=1)
            task_preds += preds.data.tolist()

        # Update metrics
        task_metrics = task.get_metrics(reset=True)
        for name, value in task_metrics.items():
            all_metrics["%s_%s" % (task.name, name)] = value
        all_metrics["micro_avg"] += all_metrics[task.val_metric] * n_examples
        all_metrics["macro_avg"] += all_metrics[task.val_metric]
        n_overall_examples += n_examples

        # Store predictions, possibly sorting them if
        if task_idxs:
            assert len(task_idxs) == len(task_preds), "Number of indices and predictions differ!"
            idxs_and_preds = [(idx, pred) for pred, idx in zip(task_idxs, task_preds)]
            idxs_and_preds.sort(key=lambda x: x[0])
            task_preds = [p for _, p in idxs_and_preds]
        all_preds[task.name] = task_preds

    all_metrics["macro_avg"] /= len(tasks)
    all_metrics["micro_avg"] /= n_overall_examples

    return all_metrics, all_preds


def write_preds(all_preds, pred_dir):
    ''' Write predictions to separate files located in pred_dir.
    We write special code to handle various GLUE tasks.

    Args:
        - all_preds (Dict[str:list]): dictionary mapping task names to predictions.
            Assumes that predictions are sorted (if necessary).
            For tasks with sentence predictions, we assume they've been mapped back to strings.
    '''
    def write_preds_to_file(preds, pred_file, pred_map=None, write_type=int):
        ''' Write preds to pred_file '''
        with open(pred_file, 'w') as pred_fh:
            pred_fh.write("index\tprediction\n")
            for idx, pred in enumerate(preds):
                if pred_map is not None or write_type == str:
                    pred = pred_map[pred]
                    pred_fh.write("%d\t%s\n" % (idx, pred))
                elif write_type == float:
                    pred_fh.write("%d\t%.3f\n" % (idx, pred))
                elif write_type == int:
                    pred_fh.write("%d\t%d\n" % (idx, pred))

    for task, preds in all_preds.items():
        if task == 'mnli': # 9796 + 9847 = 19643
            # assert len(preds) == whatever this should be
            pred_map = {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
            write_preds_to_file(preds[:9796], os.path.join(pred_dir, "%s-m.tsv" % task), pred_map)
            write_preds_to_file(preds[9796:19643], os.path.join(pred_dir, "%s-mm.tsv" % task),
                                pred_map=pred_map)
            write_preds_to_file(preds[19643:], os.path.join(pred_dir, "diagnostic.tsv"), pred_map)
        elif task in ['rte', 'qnli']:
            pred_map = {0: 'not_entailment', 1: 'entailment'}
            write_preds_to_file(preds, os.path.join(pred_dir, "%s.tsv" % task), pred_map)
        elif task in ['sts-b']:
            preds = [min(max(0., pred * 5.), 5.) for pred in preds]
            write_preds_to_file(preds, os.path.join(pred_dir, "%s.tsv" % task), write_type=float)
        elif task in ['wmt']:
            # convert each prediction to a single string if we find a list of tokens
            if isinstance(preds[0], list):
                assert isinstance(preds[0][0], str)
                preds = [' '.join(pred) for pred in preds]
            write_preds_to_file(preds, os.path.join(pred_dir, "%s.tsv" % task), write_type=str)
        else:
            write_preds_to_file(preds, os.path.join(pred_dir, "%s.tsv" % task))
    log.info("Wrote predictions to %s", pred_dir)
    return


def write_results(results, results_file, run_name):
    ''' Aggregate results by appending results to results_file '''
    with open(results_file, 'a') as results_fh:
        all_metrics_str = ', '.join(['%s: %.3f' % (metric, score) for
                                     metric, score in results.items()])
        results_fh.write("%s\t%s\n" % (run_name, all_metrics_str))
    log.info(all_metrics_str)
