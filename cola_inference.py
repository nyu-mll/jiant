"""
Run a model inference via a REPL (read-eval-print loop), or by processing an input corpus file.

To run as REPL (default):

    python cola_inference.py \
        --config_file PATH_TO_CONFIG_FILE \
        --model_file_path PATH_TO_FILE_PATH


To process a corpus:

    python cola_inference.py \
        --config_file PATH_TO_CONFIG_FILE \
        --model_file_path PATH_TO_FILE_PATH \
        --inference_mode corpus \
        --input_path PATH_TO_INPUT_CORPUS \
        --output_path PATH_TO_WRITE_OUTPUT

To process+evaluate (e.g.) the CoLA dev set:

    python cola_inference.py \
        --config_file PATH_TO_CONFIG_FILE \
        --model_file_path PATH_TO_FILE_PATH \
        --inference_mode corpus \
        --input_path PATH_TO_INPUT_CORPUS \
        --input_format dev \
        --output_path PATH_TO_WRITE_OUTPUT

(Ensure that the repository is in your PYTHONPATH when running this script.)

"""
# pylint: disable=no-member
import argparse
import json
import logging as log
import os
import sys

import numpy as np
import pandas as pd
import torch
from allennlp.data import Instance, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.nn.util import move_to_device
from tqdm import tqdm

from jiant.models import build_model
from jiant.preprocess import build_indexers, build_tasks, ModelPreprocessingInterface
from jiant.tasks.tasks import tokenize_and_truncate, sentence_to_text_field
from jiant.utils import config
from jiant.utils.data_loaders import load_tsv
from jiant.utils.utils import load_model_state, select_pool_type
from jiant.utils.options import parse_cuda_list_arg
from jiant.utils.tokenizers import select_tokenizer
from jiant.__main__ import check_arg_name

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def handle_arguments(cl_arguments):
    parser = argparse.ArgumentParser(description="")

    # Configuration files
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        nargs="+",
        help="Config file(s) (.conf) for model parameters.",
    )
    parser.add_argument(
        "--overrides",
        "-o",
        type=str,
        default=None,
        help="Parameter overrides, as valid HOCON string.",
    )

    # Inference arguments
    parser.add_argument(
        "--model_file_path", type=str, required=True, help="Path to saved model (.th)."
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="repl",
        help="Run as REPL, or process a corpus file." " [repl, corpus]",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Input path for running inference."
        " One input per line."
        " Only in eval_mode='corpus'",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="text",
        help="Format of input (text | train | dev | test)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for running inference." " Only in eval_mode='corpus'",
    )
    parser.add_argument(
        "--eval_output_path",
        type=str,
        default=None,
        help="Output path for metrics from evaluation." " Only in eval_mode='corpus'",
    )

    return parser.parse_args(cl_arguments)


def main(cl_arguments):
    """ Run REPL for a CoLA model """

    # Arguments handling #
    cl_args = handle_arguments(cl_arguments)
    args = config.params_from_file(cl_args.config_file, cl_args.overrides)
    check_arg_name(args)

    assert args.target_tasks == "cola", "Currently only supporting CoLA. ({})".format(
        args.target_tasks
    )

    if args.cuda >= 0:
        try:
            if not torch.cuda.is_available():
                raise EnvironmentError("CUDA is not available, or not detected" " by PyTorch.")
            log.info("Using GPU %d", args.cuda)
            torch.cuda.set_device(args.cuda)
        except Exception:
            log.warning(
                "GPU access failed. You might be using a CPU-only"
                " installation of PyTorch. Falling back to CPU."
            )
            args.cuda = -1

    if args.tokenizer == "auto":
        args.tokenizer = select_tokenizer(args)
    if args.pool_type == "auto":
        args.pool_type = select_pool_type(args)

    # Prepare data #
    _, target_tasks, vocab, word_embs = build_tasks(args)
    tasks = sorted(set(target_tasks), key=lambda x: x.name)

    # Build or load model #
    cuda_device = parse_cuda_list_arg(args.cuda)
    model = build_model(args, vocab, word_embs, tasks, cuda_device)
    log.info("Loading existing model from %s...", cl_args.model_file_path)
    load_model_state(model, cl_args.model_file_path, args.cuda, [], strict=False)

    # Inference Setup #
    model.eval()
    vocab = Vocabulary.from_files(os.path.join(args.exp_dir, "vocab"))
    indexers = build_indexers(args)
    task = take_one(tasks)
    model_preprocessing_interface = ModelPreprocessingInterface(args)

    # Run Inference #
    if cl_args.inference_mode == "repl":
        assert cl_args.input_path is None
        assert cl_args.output_path is None
        print("Running REPL for task: {}".format(task.name))
        run_repl(model, model_preprocessing_interface, vocab, indexers, task, args)
    elif cl_args.inference_mode == "corpus":
        run_corpus_inference(
            model,
            model_preprocessing_interface,
            vocab,
            indexers,
            task,
            args,
            cl_args.input_path,
            cl_args.input_format,
            cl_args.output_path,
            cl_args.eval_output_path,
        )
    else:
        raise KeyError(cl_args.inference_mode)


def run_repl(model, model_preprocessing_interface, vocab, indexers, task, args):
    """ Run REPL """
    print("Input CTRL-C or enter 'QUIT' to terminate.")
    while True:
        try:
            print()
            input_string = input(" INPUT: ")
            if input_string == "QUIT":
                break

            tokens = tokenize_and_truncate(
                tokenizer_name=task.tokenizer_name, sent=input_string, max_seq_len=args.max_seq_len
            )
            print("TOKENS:", " ".join("[{}]".format(tok) for tok in tokens))
            field = sentence_to_text_field(
                model_preprocessing_interface.boundary_token_fn(tokens), indexers
            )
            field.index(vocab)
            batch = Batch([Instance({"input1": field})]).as_tensor_dict()
            batch = move_to_device(batch, args.cuda)
            with torch.no_grad():
                out = model.forward(task, batch, predict=True)
            assert out["logits"].shape[1] == 2

            s = "  PRED: "
            s += "TRUE " if out["preds"][0].item() else "FALSE"
            s += "  ({:.1f}%, logits: {:.3f} vs {:.3f})".format(
                torch.softmax(out["logits"][0], dim=0)[1].item() * 100,
                out["logits"][0][0].item(),
                out["logits"][0][1].item(),
            )
            print(s)
        except KeyboardInterrupt:
            print("\nTerminating.")
            break


def run_corpus_inference(
    model,
    model_preprocessing_interface,
    vocab,
    indexers,
    task,
    args,
    input_path,
    input_format,
    output_path,
    eval_output_path,
):
    """ Run inference on corpus """
    tokens, labels = load_cola_data(input_path, task, input_format, args.max_seq_len)
    logit_batches = []
    for tokens_batch in tqdm(list(batchify(tokens, args.batch_size))):
        batch, _ = prepare_batch(model_preprocessing_interface, tokens_batch, vocab, indexers, args)
        with torch.no_grad():
            out = model.forward(task, batch, predict=True)
            logit_batches.append(out["logits"].cpu().numpy())

    logits = np.concatenate(logit_batches, axis=0)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    data_out = np.concatenate([logits, probs], axis=1)

    # Future-proofing
    assert task.name == "cola"
    num_classes = logits.shape[1]
    columns = [f"logit_{i}" for i in range(num_classes)] + [f"prob_{i}" for i in range(num_classes)]

    df = pd.DataFrame(data_out, columns=columns)
    df["pred"] = preds

    df.to_csv(output_path, index=False)

    if labels is not None:
        metrics = get_cola_metrics(logits, preds, labels, task)
        metrics_str = json.dumps(metrics, indent=2)
        print(metrics_str)
        if eval_output_path is not None:
            with open(eval_output_path, "w") as f:
                f.write(metrics_str)
    else:
        assert eval_output_path is None


def batchify(ls, batch_size):
    """ Partition a list into batches of batch_size """
    i = 0
    while i < len(ls):
        yield ls[i : i + batch_size]
        i += batch_size


def prepare_batch(model_preprocessing_interface, tokens_batch, vocab, indexers, args):
    """ Do preprocessing for batch """
    instance_ls = []
    token_ls = []
    for tokens in tokens_batch:
        field = sentence_to_text_field(
            model_preprocessing_interface.boundary_token_fn(tokens), indexers
        )
        field.index(vocab)
        instance_ls.append(Instance({"input1": field}))
        token_ls.append(tokens)
    batch = Batch(instance_ls).as_tensor_dict()
    batch = move_to_device(batch, args.cuda)
    return batch, token_ls


def take_one(ls):
    """ Extract singleton from list """
    assert len(ls) == 1
    return ls[0]


def load_cola_data(input_path, task, input_format, max_seq_len):
    if input_format == "text":
        with open(input_path, "r") as f_in:
            sentences = f_in.readlines()
        tokens = [
            tokenize_and_truncate(
                tokenizer_name=task.tokenizer_name, sent=sentence, max_seq_len=max_seq_len
            )
            for sentence in sentences
        ]
        labels = None
    elif input_format == "train" or input_format == "dev":
        data = load_tsv(
            task.tokenizer_name, input_path, max_seq_len, s1_idx=3, s2_idx=None, label_idx=1
        )
        tokens, labels = data[0], data[2]
    elif input_format == "test":
        data = load_tsv(
            task.tokenizer_name,
            input_path,
            max_seq_len,
            s1_idx=1,
            s2_idx=None,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        tokens, labels = data[0], None
    else:
        raise KeyError(input_format)
    return tokens, labels


def get_cola_metrics(logits, preds, labels, task):
    labels_tensor = torch.tensor(np.array(labels))
    logits_tensor = torch.tensor(logits)
    preds_tensor = torch.tensor(preds)
    task.scorer2(logits_tensor, labels_tensor)
    task.scorer1(labels_tensor, preds_tensor)
    return task.get_metrics(reset=True)


if __name__ == "__main__":
    main(sys.argv[1:])
