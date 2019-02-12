'''
Run a model inference via a REPL (read-eval-print loop), or by processing an input corpus file.

To run as REPL (default):

    python src/interface/inference.py \
        --config_file PATH_TO_CONFIG_FILE \
        --model_file_path PATH_TO_FILE_PATH


To process a corpus:

    python src/interface/inference.py \
        --config_file PATH_TO_CONFIG_FILE \
        --model_file_path PATH_TO_FILE_PATH \
        --inference mode corpus \
        --input_path PATH_TO_INPUT_CORPUS
        --output_path PATH_TO_WRITE_OUTPUT

(Ensure that the repository is in your PYTHONPATH when running this script.)

'''
# pylint: disable=no-member
import argparse
import numpy as np
import os
import pandas as pd
import sys

import logging as log
from tqdm import tqdm

import torch

from src.tasks.tasks import process_sentence, sentence_to_text_field
from src.utils import config
from src.utils.utils import load_model_state, check_arg_name
from src.preprocess import build_tasks, build_indexers
from src.models import build_model

from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data import Instance
from allennlp.nn.util import move_to_device

log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)


def handle_arguments(cl_arguments):
    parser = argparse.ArgumentParser(description='')

    # Configuration files
    parser.add_argument('--config_file', '-c', type=str, nargs="+",
                        help="Config file(s) (.conf) for model parameters.")
    parser.add_argument('--overrides', '-o', type=str, default=None,
                        help="Parameter overrides, as valid HOCON string.")

    # Inference arguments
    parser.add_argument('--model_file_path', type=str, required=True,
                        help="Path to saved model (.th).")
    parser.add_argument('--inference_mode', type=str, default="repl",
                        help="Run as REPL, or process a corpus file."
                             " [repl, corpus]")
    parser.add_argument('--input_path', type=str, default=None,
                        help="Input path for running inference."
                             " One input per line."
                             " Only in eval_mode='corpus'")
    parser.add_argument('--output_path', type=str, default=None,
                        help="Output path for running inference."
                             " Only in eval_mode='corpus'")

    return parser.parse_args(cl_arguments)


def main(cl_arguments):
    ''' Run REPL for a CoLA model '''

    # Arguments handling #
    cl_args = handle_arguments(cl_arguments)
    args = config.params_from_file(cl_args.config_file, cl_args.overrides)
    check_arg_name(args)
    assert args.target_tasks == "cola", \
        "Currently only supporting CoLA. ({})".format(args.target_tasks)

    if args.cuda >= 0:
        try:
            if not torch.cuda.is_available():
                raise EnvironmentError("CUDA is not available, or not detected"
                                       " by PyTorch.")
            log.info("Using GPU %d", args.cuda)
            torch.cuda.set_device(args.cuda)
        except Exception:
            log.warning(
                "GPU access failed. You might be using a CPU-only"
                " installation of PyTorch. Falling back to CPU."
            )
            args.cuda = -1

    # Prepare data #
    _, target_tasks, vocab, word_embs = build_tasks(args)
    tasks = sorted(set(target_tasks), key=lambda x: x.name)

    # Build or load model #
    model = build_model(args, vocab, word_embs, tasks)
    log.info("Loading existing model from %s...", cl_args.model_file_path)
    load_model_state(model, cl_args.model_file_path,
                     args.cuda, [], strict=False)

    # Inference Setup #
    model.eval()
    vocab = Vocabulary.from_files(os.path.join(args.exp_dir, 'vocab'))
    indexers = build_indexers(args)
    task = take_one(tasks)

    # Run Inference #
    if cl_args.inference_mode == "repl":
        assert cl_args.input_path is None
        assert cl_args.output_path is None
        print("Running REPL for task: {}".format(task.name))
        run_repl(model, vocab, indexers, task, args)
    elif cl_args.inference_mode == "corpus":
        run_corpus_inference(model, vocab, indexers, task, args,
                             cl_args.input_path, cl_args.output_path)
    else:
        raise KeyError(cl_args.inference_mode)


def run_repl(model, vocab, indexers, task, args):
    ''' Run REPL '''
    print("Input CTRL-C or enter 'QUIT' to terminate.")
    while True:
        try:
            print()
            input_string = input(" INPUT: ")
            if input_string == "QUIT":
                break

            tokens = process_sentence(
                input_string, args.max_seq_len)
            print("TOKENS:", " ".join("[{}]".format(tok) for tok in tokens))
            field = sentence_to_text_field(tokens, indexers)
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


def run_corpus_inference(model, vocab, indexers, task, args,
                         input_path, output_path):
    ''' Run inference on corpus '''
    with open(input_path, "r") as f_in:
        data_in = f_in.readlines()
    data_out_batches = []
    for batch_string_ls in tqdm(list(batchify(data_in, args.batch_size))):
        batch, _ = prepare_batch(batch_string_ls, vocab, indexers, args)
        with torch.no_grad():
            out = model.forward(task, batch, predict=True)
            data_out_batches.append(out["logits"].cpu().numpy())
    data_out = np.concatenate(data_out_batches, axis=0)
    df = pd.DataFrame(data_out)
    df.to_csv(output_path, header=False, index=False)


def batchify(ls, batch_size):
    ''' Partition a list into batches of batch_size '''
    i = 0
    while i < len(ls):
        yield ls[i:i + batch_size]
        i += batch_size


def prepare_batch(input_string_ls, vocab, indexers, args):
    ''' Do preprocessing for batch '''
    instance_ls = []
    token_ls = []
    for input_string in input_string_ls:
        tokens = process_sentence(input_string, args.max_seq_len)
        field = sentence_to_text_field(tokens, indexers)
        field.index(vocab)
        instance_ls.append(Instance({"input1": field}))
        token_ls.append(tokens)
    batch = Batch(instance_ls).as_tensor_dict()
    batch = move_to_device(batch, args.cuda)
    return batch, token_ls


def take_one(ls):
    ''' Extract singleton from list '''
    assert len(ls) == 1
    return ls[0]


if __name__ == '__main__':
    main(sys.argv[1:])
