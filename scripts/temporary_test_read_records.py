from jiant.utils import serialize
import os
import torch

frac_tasks = [
    ("edges-ner-ontonotes-5k", "edges-ner-ontonotes", 0.10059147788999316),
    ("edges-srl-ontonotes-5k", "edges-srl-ontonotes", 0.02160013824088474),
    ("edges-coref-ontonotes-5k", "edges-coref-ontonotes", 0.11968307920626182),
    ("edges-pos-ontonotes-5k", "edges-pos-ontonotes", 0.06170173381872031),
    ("edges-nonterminal-ontonotes-5k", "edges-nonterminal-ontonotes", 0.04524354600816194),
    ("edges-dep-ud-ewt-5k", "edges-dep-ud-ewt", 0.4001280409731114),
    ("se-probing-word-content-5k", "se-probing-word-content", 0.05),
    ("se-probing-tree-depth-5k", "se-probing-tree-depth", 0.05),
    ("se-probing-top-constituents-5k", "se-probing-top-constituents", 0.05),
    ("se-probing-bigram-shift-5k", "se-probing-bigram-shift", 0.05),
    ("se-probing-past-present-5k", "se-probing-past-present", 0.05),
    ("se-probing-subj-number-5k", "se-probing-subj-number", 0.05),
    ("se-probing-obj-number-5k", "se-probing-obj-number", 0.05),
    ("se-probing-odd-man-out-5k", "se-probing-odd-man-out", 0.05),
    ("se-probing-coordination-inversion-5k", "se-probing-coordination-inversion", 0.05),
    ("se-probing-sentence-length-5k", "se-probing-sentence-length", 0.0500020000800032),
    ("cola-5k", "cola", 0.584726932522512),
    ("sst-20k", "sst", 0.29696060817532555),
    ("socialiqa-20k", "socialiqa", 0.5986231667165519),
    ("ccg-20k", "ccg", 0.5261081152176772),
    ("qqp-20k", "qqp", 0.05496831076884176),
    ("mnli-20k", "mnli", 0.05092920331447255),
    ("scitail-20k", "scitail", 0.8476012883539583),
    ("qasrl-20k", "qasrl", 0.031348502873874),
    ("qamr-20k", "qamr", 0.3951397806974217),
    ("cosmosqa-20k", "cosmosqa", 0.7917029530520149),
    ("hellaswag-20k", "hellaswag", 0.5011903270266884),
    ("record-20k", "record", 0.011130552476102704),
    ("winogrande-20k", "winogrande", 0.49507401356502795),
]

for limited_size_task, task_name, frac in frac_tasks:
    filename = os.path.join(
        "/scratch/hl3236/jiant_results/",
        f"optuna_{task_name}",
        "preproc",
        f"{task_name}__train_data",
    )
    print(f"{limited_size_task} start")
    if os.path.exists(filename):
        data_old = serialize.old_read_records(filename, repeatable=False, fraction=frac)
        data_new = serialize.read_records(filename, repeatable=False, fraction=frac)
        for instance_old, instance_new in zip(data_old, data_new):
            td_old, td_new = instance_old.as_tensor_dict(), instance_new.as_tensor_dict()
            for key in td_old:
                assert repr(td_old[key]) == repr(td_new[key]), (
                    f"{limited_size_task}, {key} mismatch \n"
                    f"old: {repr(td_old[key])}m \nnew: {repr(td_new[key])}"
                )
        print(f"{limited_size_task} checked")
    else:
        print(f"{limited_size_task} data not available")
