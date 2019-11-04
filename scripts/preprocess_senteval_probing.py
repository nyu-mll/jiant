"""
 Splits senteval probing task into a train-val-test split
 Usage:
     python preprocess_senteval_probing.py --senteval_probing_path={path/to/senteval/probing/data}
"""
import os
import argparse
import pandas as pd


def parse_senteval_probing(args):
    files = [x for x in os.listdir(args.senteval_probing_path) if "txt" in x]
    for file in files:
        file_pd = pd.read_fwf(os.path.join(args.senteval_probing_path, file), header=None)
        files_train = file_pd[file_pd[0] == "tr"]
        task_name = file.split(".")[0]
        if not os.path.exists(os.path.join(args.senteval_probing_path, task_name)):
            os.mkdir(os.path.join(args.senteval_probing_path, task_name))
        files_train.to_csv(os.path.join(args.senteval_probing_path, task_name, "train.csv"))
        files_val = file_pd[file_pd[0] == "va"]
        task_name = file.split(".")[0]
        files_val.to_csv(os.path.join(args.senteval_probing_path, task_name, "val.csv"))

        files_test = file_pd[file_pd[0] == "te"]
        task_name = file.split(".")[0]
        files_test.to_csv(os.path.join(args.senteval_probing_path, task_name, "test.csv"))


parser = argparse.ArgumentParser()
parser.add_argument("--senteval_probing_path", type=str, help="path to original Senteval files")

args = parser.parse_args()
parse_senteval_probing(args)
