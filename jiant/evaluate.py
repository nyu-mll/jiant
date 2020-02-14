""" Helper functions to evaluate a model on a dataset """
import json
import logging as log
import os
import time
from collections import defaultdict
from csv import QUOTE_MINIMAL, QUOTE_NONE
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
from allennlp.nn.util import move_to_device
from allennlp.data.iterators import BasicIterator
from jiant import tasks as tasks_module
from jiant.tasks.tasks import (
    BooleanQuestionTask,
    CommitmentTask,
    COPATask,
    RTESuperGLUETask,
    WiCTask,
    WinogradCoreferenceTask,
    GLUEDiagnosticTask,
)
from jiant.tasks.qa import MultiRCTask, ReCoRDTask, QASRLTask
from jiant.tasks.edge_probing import EdgeProbingTask
from jiant.utils.utils import get_output_attribute


LOG_INTERVAL = 30


def _format_preds(preds):
    if isinstance(preds, (list, torch.Tensor)):
        preds = _coerce_list(preds)
        assert isinstance(preds, list), "Convert predictions to list!"
        cols = {"preds": preds}
    elif isinstance(preds, dict):
        cols = {}
        for k, v in preds.items():
            cols[f"preds_{k}"] = _coerce_list(v)
    else:
        raise TypeError(type(preds))
    return cols


def _coerce_list(preds) -> List:
    if isinstance(preds, torch.Tensor):
        return preds.data.tolist()
    else:
        return list(preds)


def parse_write_preds_arg(write_preds_arg: str) -> List[str]:
    if write_preds_arg == 0:
        return []
    elif write_preds_arg == 1:
        return ["test"]
    else:
        return write_preds_arg.split(",")


def evaluate(
    model, tasks: Sequence[tasks_module.Task], batch_size: int, cuda_device, split="val"
) -> Tuple[Dict, pd.DataFrame]:
    """Evaluate on a dataset
    {par,qst,ans}_idx are used for MultiRC and other question answering dataset"""
    FIELDS_TO_EXPORT = [
        "idx",
        "sent1_str",
        "sent2_str",
        "labels",
        "pair_id",
        "psg_idx",
        "qst_idx",
        "ans_idx",
        "ans_str",
    ]
    # Enforce that these tasks have the 'idx' field set.
    IDX_REQUIRED_TASK_NAMES = (
        tasks_module.ALL_GLUE_TASKS
        + tasks_module.ALL_SUPERGLUE_TASKS
        + tasks_module.ALL_COLA_NPI_TASKS
    )
    model.eval()
    iterator = BasicIterator(batch_size)

    all_metrics = {"micro_avg": 0.0, "macro_avg": 0.0}
    all_preds = {}
    n_examples_overall = 0  # n examples over all tasks
    assert len(tasks) > 0, "Configured to evaluate, but specified no task to evaluate."

    for task in tasks:
        log.info("Evaluating on: %s, split: %s", task.name, split)
        last_log = time.time()
        n_task_examples = 0
        task_preds = []  # accumulate DataFrames
        assert split in ["train", "val", "test"]
        dataset = getattr(task, "%s_data" % split)
        generator = iterator(dataset, num_epochs=1, shuffle=False)
        for batch_idx, batch in enumerate(generator):
            with torch.no_grad():
                if isinstance(cuda_device, int):
                    batch = move_to_device(batch, cuda_device)
                out = model.forward(task, batch, predict=True)
            if task is not None:
                task.update_metrics(out, batch)
            n_task_examples += get_output_attribute(out, "n_exs", cuda_device)
            # get predictions
            if "preds" not in out:
                continue
            out["preds"] = task.handle_preds(out["preds"], batch)
            cols = _format_preds(out["preds"])
            if task.name in IDX_REQUIRED_TASK_NAMES:
                assert "idx" in batch, f"'idx' field missing from batches " "for task {task.name}!"
            for field in FIELDS_TO_EXPORT:
                if field in batch:
                    cols[field] = _coerce_list(batch[field])

            # Transpose data using Pandas
            df = pd.DataFrame(cols)
            task_preds.append(df)

            if time.time() - last_log > LOG_INTERVAL:
                log.info("\tTask %s: batch %d", task.name, batch_idx)
                last_log = time.time()
        # task_preds will be a DataFrame with columns
        # ['preds'] + FIELDS_TO_EXPORT
        # for GLUE tasks, preds entries should be single scalars.
        # Update metrics
        task_metrics = task.get_metrics(reset=True)
        for name, value in task_metrics.items():
            all_metrics["%s_%s" % (task.name, name)] = value

        # We don't want diagnostic tasks to affect the micro and macro average.
        # Accuracy on diagnostic tasks is hardcoded to 0 except for winogender-diagnostic.
        if task.contributes_to_aggregate_score:
            all_metrics["micro_avg"] += all_metrics[task.val_metric] * n_task_examples
            all_metrics["macro_avg"] += all_metrics[task.val_metric]
            n_examples_overall += n_task_examples

        if not task_preds:
            log.warning("Task %s: has no predictions!", task.name)
            continue

        # Combine task_preds from each batch to a single DataFrame.
        task_preds = pd.concat(task_preds, ignore_index=True)

        # Store predictions, sorting by index if given.
        if "idx" in task_preds.columns:
            log.info("Task '%s': sorting predictions by 'idx'", task.name)
            task_preds.sort_values(by=["idx"], inplace=True)
        all_preds[task.name] = task_preds
        log.info("Finished evaluating on: %s", task.name)

    # hack for diagnostics
    all_metrics["micro_avg"] /= max(n_examples_overall, 1)
    all_metrics["macro_avg"] /= len(tasks)

    return all_metrics, all_preds


def write_preds(
    tasks: Iterable[tasks_module.Task], all_preds, pred_dir, split_name, strict_glue_format=False
) -> None:
    for task in tasks:
        if task.name not in all_preds:
            log.warning("Task '%s': missing predictions for split '%s'", task.name, split_name)
            continue

        preds_df = all_preds[task.name]
        # Tasks that use _write_glue_preds:
        glue_style_tasks = (
            tasks_module.ALL_NLI_PROBING_TASKS
            + tasks_module.ALL_GLUE_TASKS
            + ["wmt"]
            + tasks_module.ALL_COLA_NPI_TASKS
        )

        if task.name in glue_style_tasks:
            # Strict mode: strict GLUE format (no extra cols)
            strict = strict_glue_format and task.name in tasks_module.ALL_GLUE_TASKS
            _write_glue_preds(task.name, preds_df, pred_dir, split_name, strict_glue_format=strict)
        elif isinstance(task, EdgeProbingTask):
            # Edge probing tasks, have structured output.
            _write_edge_preds(task, preds_df, pred_dir, split_name)
        elif isinstance(task, BooleanQuestionTask):
            _write_boolq_preds(
                task, preds_df, pred_dir, split_name, strict_glue_format=strict_glue_format
            )
        elif isinstance(task, CommitmentTask):
            _write_commitment_preds(
                task, preds_df, pred_dir, split_name, strict_glue_format=strict_glue_format
            )
        elif isinstance(task, COPATask):
            _write_copa_preds(
                task, preds_df, pred_dir, split_name, strict_glue_format=strict_glue_format
            )
        elif isinstance(task, MultiRCTask):
            _write_multirc_preds(
                task, preds_df, pred_dir, split_name, strict_glue_format=strict_glue_format
            )
        elif isinstance(task, RTESuperGLUETask):
            _write_rte_preds(
                task, preds_df, pred_dir, split_name, strict_glue_format=strict_glue_format
            )
        elif isinstance(task, ReCoRDTask):
            _write_record_preds(
                task, preds_df, pred_dir, split_name, strict_glue_format=strict_glue_format
            )
        elif isinstance(task, WiCTask):
            _write_wic_preds(
                task, preds_df, pred_dir, split_name, strict_glue_format=strict_glue_format
            )
        elif isinstance(task, WinogradCoreferenceTask):
            _write_winograd_preds(
                task, preds_df, pred_dir, split_name, strict_glue_format=strict_glue_format
            )
        elif isinstance(task, GLUEDiagnosticTask):
            # glue-diagnostic is caught above by being in ALL_GLUE_TASKS
            # currently this only catches superglue-diagnostic
            _write_diagnostics_preds(
                task, preds_df, pred_dir, split_name, strict_glue_format=strict_glue_format
            )
        elif isinstance(task, QASRLTask):
            _write_simple_tsv_preds(task, preds_df, pred_dir, split_name)
        else:
            log.warning("Task '%s' not supported by write_preds().", task.name)
            continue
        log.info("Task '%s': Wrote predictions to %s", task.name, pred_dir)
    log.info("Wrote all preds for split '%s' to %s", split_name, pred_dir)
    return


# Exact file names per task required by the GLUE evaluation server
GLUE_NAME_MAP = {
    "cola": "CoLA",
    "glue-diagnostic": "AX",
    "mnli-mm": "MNLI-mm",
    "mnli-m": "MNLI-m",
    "mrpc": "MRPC",
    "qnli": "QNLI",
    "qqp": "QQP",
    "rte": "RTE",
    "sst": "SST-2",
    "sts-b": "STS-B",
    "wnli": "WNLI",
}

# Exact file names per task required by the SuperGLUE evaluation server
SUPERGLUE_NAME_MAP = {
    "boolq": "BoolQ",
    "commitbank": "CB",
    "copa": "COPA",
    "multirc": "MultiRC",
    "record": "ReCoRD",
    "rte-superglue": "RTE",
    "wic": "WiC",
    "winograd-coreference": "WSC",
    "broadcoverage-diagnostic": "AX-b",
    "winogender-diagnostic": "AX-g",
}


def _get_pred_filename(task_name, pred_dir, split_name, strict_glue_format):
    if strict_glue_format and task_name in GLUE_NAME_MAP:
        if split_name == "test":
            file = "%s.tsv" % (GLUE_NAME_MAP[task_name])
        else:
            file = "%s_%s.tsv" % (GLUE_NAME_MAP[task_name], split_name)
    elif strict_glue_format and task_name in SUPERGLUE_NAME_MAP:
        if split_name == "test":
            file = "%s.jsonl" % (SUPERGLUE_NAME_MAP[task_name])
        else:
            file = "%s_%s.jsonl" % (SUPERGLUE_NAME_MAP[task_name], split_name)
    else:
        file = "%s_%s.tsv" % (task_name, split_name)
    return os.path.join(pred_dir, file)


def _write_edge_preds(
    task: EdgeProbingTask,
    preds_df: pd.DataFrame,
    pred_dir: str,
    split_name: str,
    join_with_input: bool = True,
):
    """ Write predictions for edge probing task.

    This reads the task data and joins with predictions,
    taking the 'idx' field to represent the line number in the (preprocessed)
    task data file.

    Predictions are saved as JSON with one record per line.
    """
    preds_file = os.path.join(pred_dir, f"{task.name}_{split_name}.json")
    # Each row of 'preds' is a NumPy object, need to convert to list for
    # serialization.
    preds_df = preds_df.copy()
    preds_df["preds"] = [a.tolist() for a in preds_df["preds"]]
    if join_with_input:
        preds_df.set_index(["idx"], inplace=True)
        # Load input data and join by row index.
        log.info("Task '%s': joining predictions with input split '%s'", task.name, split_name)
        records = task.get_split_text(split_name)
        # TODO: update this with more prediction types, when available.
        records = (
            task.merge_preds(r, {"proba": preds_df.at[i, "preds"]}) for i, r in enumerate(records)
        )
    else:
        records = (row.to_dict() for _, row in preds_df.iterrows())

    with open(preds_file, "w") as fd:
        for record in records:
            fd.write(json.dumps(record))
            fd.write("\n")


def _write_wic_preds(
    task: str,
    preds_df: pd.DataFrame,
    pred_dir: str,
    split_name: str,
    strict_glue_format: bool = False,
):
    """ Write predictions for WiC task.  """
    pred_map = {0: "false", 1: "true"}
    preds_file = _get_pred_filename(task.name, pred_dir, split_name, strict_glue_format)
    with open(preds_file, "w", encoding="utf-8") as preds_fh:
        for row_idx, row in preds_df.iterrows():
            if strict_glue_format:
                out_d = {"idx": row["idx"], "label": pred_map[row["preds"]]}
            else:
                out_d = row.to_dict()
            preds_fh.write("{0}\n".format(json.dumps(out_d)))


def _write_winograd_preds(
    task: str,
    preds_df: pd.DataFrame,
    pred_dir: str,
    split_name: str,
    strict_glue_format: bool = False,
):
    """ Write predictions for Winograd Coreference task.  """
    pred_map = {0: "False", 1: "True"}
    preds_file = _get_pred_filename(task.name, pred_dir, split_name, strict_glue_format)
    with open(preds_file, "w", encoding="utf-8") as preds_fh:
        for row_idx, row in preds_df.iterrows():
            if strict_glue_format:
                out_d = {"idx": int(row["idx"]), "label": pred_map[row["preds"]]}
            else:
                out_d = row.to_dict()
            preds_fh.write("{0}\n".format(json.dumps(out_d)))


def _write_boolq_preds(
    task: str,
    preds_df: pd.DataFrame,
    pred_dir: str,
    split_name: str,
    strict_glue_format: bool = False,
):
    """ Write predictions for Boolean Questions task.  """
    pred_map = {0: "false", 1: "true"}
    preds_file = _get_pred_filename(task.name, pred_dir, split_name, strict_glue_format)
    with open(preds_file, "w", encoding="utf-8") as preds_fh:
        for row_idx, row in preds_df.iterrows():
            if strict_glue_format:
                out_d = {"idx": int(row["idx"]), "label": pred_map[row["preds"]]}
            else:
                out_d = row.to_dict()
            preds_fh.write("{0}\n".format(json.dumps(out_d)))


def _write_commitment_preds(
    task: str,
    preds_df: pd.DataFrame,
    pred_dir: str,
    split_name: str,
    strict_glue_format: bool = False,
):
    """ Write predictions for CommitmentBank task.  """
    pred_map = {0: "neutral", 1: "entailment", 2: "contradiction"}
    preds_file = _get_pred_filename(task.name, pred_dir, split_name, strict_glue_format)
    with open(preds_file, "w", encoding="utf-8") as preds_fh:
        for row_idx, row in preds_df.iterrows():
            if strict_glue_format:
                out_d = {"idx": row["idx"], "label": pred_map[row["preds"]]}
            else:
                out_d = row.to_dict()
            preds_fh.write("{0}\n".format(json.dumps(out_d)))


def _write_copa_preds(
    task, preds_df: pd.DataFrame, pred_dir: str, split_name: str, strict_glue_format: bool = False
):
    """ Write COPA predictions to JSONL """
    preds_file = _get_pred_filename(task.name, pred_dir, split_name, strict_glue_format)
    with open(preds_file, "w", encoding="utf-8") as preds_fh:
        for row_idx, row in preds_df.iterrows():
            if strict_glue_format:
                out_d = {"idx": int(row["idx"]), "label": int(row["preds"])}
            else:
                out_d = row.to_dict()
            preds_fh.write("{0}\n".format(json.dumps(out_d)))


def _write_multirc_preds(
    task: str,
    preds_df: pd.DataFrame,
    pred_dir: str,
    split_name: str,
    strict_glue_format: bool = False,
):
    """ Write predictions for MultiRC task. """
    preds_file = _get_pred_filename(task.name, pred_dir, split_name, strict_glue_format)
    with open(preds_file, "w", encoding="utf-8") as preds_fh:
        if strict_glue_format:
            par_qst_ans_d = defaultdict(lambda: defaultdict(list))
            for row_idx, row in preds_df.iterrows():
                ans_d = {"idx": int(row["ans_idx"]), "label": int(row["preds"])}
                par_qst_ans_d[int(row["psg_idx"])][int(row["qst_idx"])].append(ans_d)
            for par_idx, qst_ans_d in par_qst_ans_d.items():
                qst_ds = []
                for qst_idx, answers in qst_ans_d.items():
                    qst_d = {"idx": qst_idx, "answers": answers}
                    qst_ds.append(qst_d)
                out_d = {"idx": par_idx, "passage": {"questions": qst_ds}}
                preds_fh.write("{0}\n".format(json.dumps(out_d)))
        else:
            for row_idx, row in preds_df.iterrows():
                out_d = row.to_dict()
                preds_fh.write("{0}\n".format(json.dumps(out_d)))


def _write_record_preds(
    task: str,
    preds_df: pd.DataFrame,
    pred_dir: str,
    split_name: str,
    strict_glue_format: bool = False,
):
    """ Write predictions for ReCoRD task. """
    preds_file = _get_pred_filename(task.name, pred_dir, split_name, strict_glue_format)
    with open(preds_file, "w", encoding="utf-8") as preds_fh:
        if strict_glue_format:
            par_qst_ans_d = defaultdict(lambda: defaultdict(list))
            for row_idx, row in preds_df.iterrows():
                ans_d = {
                    "idx": int(row["ans_idx"]),
                    "str": row["ans_str"],
                    "logit": torch.FloatTensor(row["preds"]),
                }
                par_qst_ans_d[row["psg_idx"]][row["qst_idx"]].append(ans_d)
            for par_idx, qst_ans_d in par_qst_ans_d.items():
                for qst_idx, ans_ds in qst_ans_d.items():

                    # get prediction
                    logits_and_anss = [(d["logit"], d["str"]) for d in ans_ds]
                    logits_and_anss.sort(key=lambda x: x[1])
                    logits, anss = list(zip(*logits_and_anss))
                    pred_idx = torch.softmax(torch.stack(logits), dim=-1)[:, -1].argmax().item()
                    answer = anss[pred_idx]

                    # write out answer
                    qst_d = {"idx": qst_idx, "label": answer}
                    preds_fh.write("{0}\n".format(json.dumps(qst_d)))
        else:
            for row_idx, row in preds_df.iterrows():
                out_d = row.to_dict()
                preds_fh.write("{0}\n".format(json.dumps(out_d)))


def _write_rte_preds(
    task: str,
    preds_df: pd.DataFrame,
    pred_dir: str,
    split_name: str,
    strict_glue_format: bool = False,
):
    """ Write predictions for RTE task in SuperGLUE prediction format.  """
    trg_map = {0: "not_entailment", 1: "entailment"}
    preds_file = _get_pred_filename(task.name, pred_dir, split_name, strict_glue_format)
    with open(preds_file, "w", encoding="utf-8") as preds_fh:
        for row_idx, row in preds_df.iterrows():
            if strict_glue_format:
                out_d = {"idx": row["idx"], "label": trg_map[row["preds"]]}
            else:
                out_d = row.to_dict()
            preds_fh.write("{0}\n".format(json.dumps(out_d)))


def _write_simple_tsv_preds(task, preds_df: pd.DataFrame, pred_dir: str, split_name: str):
    preds_file = _get_pred_filename(task.name, pred_dir, split_name, strict_glue_format=False)
    preds_df.to_csv(preds_file, sep="\t")


def _write_diagnostics_preds(
    task: str,
    preds_df: pd.DataFrame,
    pred_dir: str,
    split_name: str,
    strict_glue_format: bool = False,
):
    """ Write predictions for GLUE/SuperGLUE diagnostics task.  """

    if task.n_classes == 2:
        pred_map = {0: "not_entailment", 1: "entailment"}
    elif task.n_classes == 3:
        pred_map = {0: "neutral", 1: "entailment", 2: "contradiction"}
    else:
        raise ValueError("Invalid number of output classes detected")

    preds_file = _get_pred_filename(task.name, pred_dir, split_name, strict_glue_format)
    with open(preds_file, "w", encoding="utf-8") as preds_fh:
        for row_idx, row in preds_df.iterrows():
            if strict_glue_format:
                out_d = {"idx": row["idx"], "label": pred_map[row["preds"]]}
            else:
                out_d = row.to_dict()
            preds_fh.write("{0}\n".format(json.dumps(out_d)))


def _write_glue_preds(
    task_name: str,
    preds_df: pd.DataFrame,
    pred_dir: str,
    split_name: str,
    strict_glue_format: bool = False,
):
    """ Write predictions to separate files located in pred_dir.
    We write special code to handle various GLUE tasks.

    Use strict_glue_format to guarantee compatibility with GLUE website.

    Args:
        task_name: task name
        preds_df: predictions DataFrame for a single task, as returned by
            evaluate().
        pred_dir: directory to write predictions
        split_name: name of this split ('train', 'val', or 'test')
        strict_glue_format: if true, writes format compatible with GLUE
            website.
    """

    def _apply_pred_map(preds_df, pred_map, key="prediction"):
        """ Apply preds_map, in-place. """
        preds_df[key] = [pred_map[p] for p in preds_df[key]]

    def _write_preds_with_pd(preds_df: pd.DataFrame, pred_file: str, write_type=int):
        """ Write TSV file in GLUE format, using Pandas. """

        required_cols = ["index", "prediction"]
        if strict_glue_format:
            cols_to_write = required_cols
            quoting = QUOTE_NONE
            log.info(
                "Task '%s', split '%s': writing %s in " "strict GLUE format.",
                task_name,
                split_name,
                pred_file,
            )
        else:
            all_cols = set(preds_df.columns)
            # make sure we write index and prediction as first columns,
            # then all the other ones we can find.
            cols_to_write = required_cols + sorted(list(all_cols.difference(required_cols)))
            quoting = QUOTE_MINIMAL
        preds_df.to_csv(
            pred_file,
            sep="\t",
            index=False,
            float_format="%.3f",
            quoting=quoting,
            columns=cols_to_write,
        )

    if len(preds_df) == 0:  # catch empty lists
        log.warning("Task '%s': predictions are empty!", task_name)
        return

    def _add_default_column(df, name: str, val):
        """ Ensure column exists and missing values = val. """
        if name not in df:
            df[name] = val
        df[name].fillna(value=val, inplace=True)

    preds_df = preds_df.copy()
    _add_default_column(preds_df, "idx", -1)
    _add_default_column(preds_df, "sent1_str", "")
    _add_default_column(preds_df, "sent2_str", "")
    _add_default_column(preds_df, "labels", -1)
    # Rename columns to match output headers.
    preds_df.rename(
        {
            "idx": "index",
            "preds": "prediction",
            "sent1_str": "sentence_1",
            "sent2_str": "sentence_2",
            "labels": "true_label",
        },
        axis="columns",
        inplace=True,
    )

    if task_name == "mnli" and split_name == "test":  # 9796 + 9847 = 19643
        assert len(preds_df) == 19643, "Missing predictions for MNLI!"
        log.info("There are %d examples in MNLI, 19643 were expected", len(preds_df))
        # Sort back to original order to split matched and mismatched, which are
        # treated as a single dataset by jiant.
        preds_df.sort_index(inplace=True)
        pred_map = {0: "neutral", 1: "entailment", 2: "contradiction"}
        _apply_pred_map(preds_df, pred_map, "prediction")
        _write_preds_with_pd(
            preds_df.iloc[:9796],
            _get_pred_filename("mnli-m", pred_dir, split_name, strict_glue_format),
        )
        _write_preds_with_pd(
            preds_df.iloc[9796:],
            _get_pred_filename("mnli-mm", pred_dir, split_name, strict_glue_format),
        )
    elif task_name in ["rte", "qnli"]:
        pred_map = {0: "not_entailment", 1: "entailment"}
        _apply_pred_map(preds_df, pred_map, "prediction")
        _write_preds_with_pd(
            preds_df, _get_pred_filename(task_name, pred_dir, split_name, strict_glue_format)
        )
    elif task_name in ["sts-b"]:
        preds_df["prediction"] = [min(max(0.0, pred * 5.0), 5.0) for pred in preds_df["prediction"]]
        _write_preds_with_pd(
            preds_df,
            _get_pred_filename(task_name, pred_dir, split_name, strict_glue_format),
            write_type=float,
        )
    elif task_name in ["wmt"]:
        # convert each prediction to a single string if we find a list of
        # tokens
        if isinstance(preds_df["prediction"][0], list):
            assert isinstance(preds_df["prediction"][0][0], str)
            preds_df["prediction"] = [" ".join(pred) for pred in preds_df["prediction"]]
        _write_preds_with_pd(
            preds_df,
            _get_pred_filename(task_name, pred_dir, split_name, strict_glue_format),
            write_type=str,
        )
    else:
        _write_preds_with_pd(
            preds_df,
            _get_pred_filename(task_name, pred_dir, split_name, strict_glue_format),
            write_type=int,
        )

    log.info("Wrote predictions for task: %s", task_name)


def write_results(results, results_file, run_name):
    """ Aggregate results by appending results to results_file """
    all_metrics_str = ", ".join(["%s: %.3f" % (metric, score) for metric, score in results.items()])
    with open(results_file, "a") as results_fh:
        results_fh.write("%s\t%s\n" % (run_name, all_metrics_str))
    log.info(all_metrics_str)
