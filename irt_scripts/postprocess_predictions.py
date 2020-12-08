import torch
import numpy as np
import glob
import csv
import json
import sys
from jiant.tasks.lib.templates.shared import labels_to_bimap


working_dir = sys.argv[1]


def write_responder_acc(responder_acc, task):
    pair_ids = []
    pair_ids = [task + "_" + str(idx) for idx in range(len(responder_acc))]
    predictions = dict(zip(pair_ids, responder_acc))
    return predictions



def write_irt_coded_predictions(predictions, data_dict_list, task):
    with open(working_dir+"experiments/irt_csv_files/" + task + "_irt_all_coded.csv", 'w') as f:
        header = ['userid']
        header.extend([*predictions][:-1])
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(data_dict_list)

def process_task(task):
    data_dict_list = []
    model_paths =  glob.glob(working_dir+'/experiments/predict_files/*/' + task + '_config*.p/test_preds.p')
    ckpt_percents = ["1", "10", "25", "50", 'best']
   
    ckpt_index_steps = []
    for model in model_paths:
        ckpt_index = model.split('/')[-2].split('_')[-1][:-2]

        if ckpt_index != "model" and int(ckpt_index) not in ckpt_index_steps:
            ckpt_index_steps.append(int(ckpt_index))
    ckpt_index_steps.sort()
    ckpt_index_steps.append("model")
    ckpt_map = dict(zip(ckpt_index_steps, ckpt_percents))

    responder_accuracies = None
    for output_file in model_paths:
        model_type = output_file.split('/')[-3]
        ckpt_index = output_file.split('/')[-2].split('_')[-1][:-2]
        if ckpt_index != "model":
            ckpt_index = ckpt_map[int(ckpt_index)]
        else:
            ckpt_index = ckpt_map[ckpt_index]
        try:
            test_dict = torch.load(output_file)[task]
            if "responder_accuracies" in test_dict:
                responder_accuracies = test_dict["responder_accuracies"]
            test_preds = test_dict['preds']
        except:
            print("error: empty preds ", output_file)
            continue

        if responder_accuracies is not None:
            predictions = write_responder_acc(responder_accuracies, task)
        
        predictions['userid'] = model_type + "_" + ckpt_index
        data_dict_list.append(predictions)

    write_irt_coded_predictions(predictions, data_dict_list, task)


if __name__ == "__main__":
    tasks = "boolq cb commonsenseqa copa rte wic snli qamr cosmosqa hellaswag wsc socialiqa arc_challenge arc_easy squad_v2 arct mnli piqa mutual mutual_plus quoref mrqa_natural_questions newsqa mcscript mctaco quail winogrande abductive_nli".split(' ')

    for task in tasks:
        process_task(task)
        