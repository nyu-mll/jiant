# Create a set of kubernetes run commands to launch the full set of
# experiments used in the paper.

# Used to generate 191223-paper-runs.sh, which is meant to run on a Google Cloud
# Platform Kubernetes cluster set up according to the instructions in gcp/kubernetes.

# Results files can be converted to LaTeX tables using results_to_table.py

BASENAME = "191223"  # Short descriptor of the full experiment.
EXP_PATH = "/nfs/jiant/exp/srbowman/srbowman"  # Base path run directories.
# Should match the jiant `project_dir` config option.

# Which tasks to pretrain on in each intermediate training setup
pretrain_task_lists = {
    "00": "none",  # No intermediate training.
    "a9": "nli-a",  # Intermediate training on our new data.
    "b9": "nli-b",  # Intermediate training on our new data.
    "c9": "nli-c",  # Intermediate training on our new data.
    "d9": "nli-d",  # Intermediate training on our new data.
    "f9": "nli-f",  # Intermediate training on our new data.
    "m9": "mnli",  # Intermediate training on a sample from MNLI.
    "g9": "mnli-government",  # Intermediate training on a single-genre sample from MNLI.
    "ma": "mnli",  # Intermediate training on MNLI.
    "x9": '\\"snli,mnli,adversarial_nli\\"',  # Intermediate training on a sample from adversarial_nli (following standard practice by adding SNLI/MNLI for training).
    "xa": '\\"snli,mnli,adversarial_nli\\"',  # Intermediate training on adversarial_nli (following standard practice by adding SNLI/MNLI for training).
}

# Whether to subsample the training data in each intermediate/hypothesis-only
# training setup, and by how much
pretrain_data_fractions = {
    "00": "1.0",
    "a9": "1.0",
    "b9": "1.0",
    "c9": "1.0",
    "d9": "1.0",
    "f9": "1.0",
    "m9": "0.021644",  # 8500/392703
    "g9": "0.109885",  # 8500/77353
    "ma": "1.0",
    "x9": "0.007687",  # 8500/(550152+392703+162765)
    "xa": "1.0",
}

# Which tasks to evaluate on in each job. SuperGLUE is split into four sections
# to speed up the return of results—not for any deep reason.
target_task_lists = {
    "pre": '\\"mnli,mnli-two\\"',  # Simplifies setup for intrinsic evaluations - not directly used.
    "sgl": '\\"winograd-coreference,copa,commitbank\\"',  # SuperGLUE evaluation job.
    "sgs": '\\"boolq,multirc,rte-superglue,wic\\"',  # SuperGLUE evaluation job.
    "sgr": "record",  # SuperGLUE evaluation job.
    "sgt": '\\"rte-superglue,broadcoverage-diagnostic,winogender-diagnostic\\"',  # SuperGLUE evaluation job.
    "nli": None,  # NLI evaluation job - depends on setting - logic below.
    "dia": '\\"mnli,glue-diagnostic\\"',  # NLI evaluation job - depends on setting - logic below.
    "han": '\\"hans\\"',  # NLI evaluation job - depends on setting - logic below.
    "sgd": '\\"broadcoverage-diagnostic\\"',
}

models = {"xlnet": "xlnet-large-cased", "roberta": "roberta-large"}

# Set up transfer learning experiments
for config in ["pre", "sgl", "sgs", "sgr", "sgt"]:
    target_override = target_task_lists[config]
    do_pretrain_override = "1" if config == "pre" else "0"
    do_target_task_training_override = "0" if config == "pre" else "1"
    do_full_eval_override = "0" if config == "pre" else "1"

    # Ensure we save checkpoints - these runs are fairly fast
    script = (
        "nli_data_evaluation_internal.conf"
        if config == "pre"
        else "nli_data_evaluation_internal_local.conf"
    )

    for model in ["roberta", "xlnet"]:
        model_override = models[model]

        for pretrain in ["00", "a9", "b9", "c9", "d9", "f9", "m9", "g9", "ma", "x9", "xa"]:
            # No need for an intermediate training job in the no-intermediate-task setting
            if pretrain == "00" and config == "pre":
                continue

            pretrain_override = pretrain_task_lists[pretrain]
            data_fraction_override = pretrain_data_fractions[pretrain]

            for restart in range(3):

                if config == "pre" or pretrain == "00":
                    load_target_train_checkpoint_override = ""
                else:
                    pre_runname = f"{BASENAME}-pre-{model}-{pretrain}-{restart}"
                    load_target_train_checkpoint_override = f"load_target_train_checkpoint = /nfs/jiant/exp/srbowman/srbowman/{BASENAME}-{model}/{pre_runname}/model*best*, "

                runname = f"{BASENAME}-{config}-{model}-{pretrain}-{restart}"
                command = (
                    f"export RUNNAME={runname}; "
                    f'gcp/kubernetes/run_batch.sh $RUNNAME "python $JIANT_PATH/main.py '
                    f"--config_file $JIANT_PATH/jiant/config/{script} "
                    f"--overrides 'exp_name = {BASENAME}-{model}, run_name = $RUNNAME, "
                    f"pretrain_tasks = {pretrain_override}, target_tasks = {target_override}, "
                    f"do_pretrain = {do_pretrain_override}, input_module = {model_override}, "
                    f"pretrain_data_fraction = {data_fraction_override}, {load_target_train_checkpoint_override}"
                    f"do_target_task_training = {do_target_task_training_override}, "
                    f"do_full_eval = {do_full_eval_override}'\""
                )
                print(command)
        print("")
    print("")
print("")


# If pretraining on some task, how should we evaluate its NLI performance
nli_evaluation_settings = {
    "a9": '\\"hans,mnli,nli-a\\"',  # Intermediate training on our new data.
    "b9": '\\"hans,mnli,nli-b\\"',  # Intermediate training on our new data.
    "c9": '\\"hans,mnli,nli-c\\"',  # Intermediate training on our new data.
    "d9": '\\"hans,mnli,nli-d\\"',  # Intermediate training on our new data.
    "f9": '\\"hans,mnli-two,nli-f,mnli\\"',  # Intermediate training on our new data.
    "m9": '\\"hans,mnli,mnli-two\\"',  # Intermediate training on a sample from MNLI.
    "g9": '\\"hans,mnli,mnli-government\\"',  # Intermediate training on a single-genre sample from MNLI.
    "ma": '\\"hans,mnli,mnli-two\\"',  # Intermediate training on MNLI.
    "x9": '\\"hans,snli,mnli,adversarial_nli\\"',  # Intermediate training on a sample from adversarial_nli (following standard practice by adding SNLI/MNLI for training).
    "xa": '\\"hans,snli,mnli,adversarial_nli\\"',  # Intermediate training on adversarial_nli (following standard practice by adding SNLI/MNLI for training).
}

# Set up NLI intrinsic evaluations. These depend on the above.
for config in ["nli", "dia", "han", "sgd"]:
    do_pretrain_override = "0"
    do_target_task_training_override = "0"
    do_full_eval_override = "1"

    script = "nli_data_evaluation_internal.conf"

    for model in ["roberta", "xlnet"]:
        model_override = models[model]

        for pretrain in ["a9", "b9", "c9", "d9", "f9", "m9", "g9", "ma", "x9", "xa"]:
            pretrain_override = "mnli"
            data_fraction_override = pretrain_data_fractions[pretrain]
            if config == "nli":
                target_override = nli_evaluation_settings[pretrain]
            else:
                target_override = target_task_lists[config]

            for restart in range(3):
                pre_runname = f"{BASENAME}-pre-{model}-{pretrain}-{restart}"
                load_target_train_checkpoint_override = f"load_eval_checkpoint = {EXP_PATH}/{BASENAME}-{model}/{pre_runname}/model*best*, "

                runname = f"{BASENAME}-{config}-{model}-{pretrain}-{restart}"
                command = (
                    f"export RUNNAME={runname}; "
                    f'gcp/kubernetes/run_batch.sh $RUNNAME "python $JIANT_PATH/main.py '
                    f"--config_file $JIANT_PATH/jiant/config/{script} "
                    f"--overrides 'exp_name = {BASENAME}-{model}, run_name = $RUNNAME, "
                    f"pretrain_tasks = {pretrain_override}, target_tasks = {target_override}, "
                    f"do_pretrain = {do_pretrain_override}, input_module = {model_override}, "
                    f"pretrain_data_fraction = {data_fraction_override}, {load_target_train_checkpoint_override}"
                    f"do_target_task_training = {do_target_task_training_override}, "
                    f"do_full_eval = {do_full_eval_override}'\""
                )
                print(command)
        print("")
    print("")
print("")

# Which tasks to pretrain on in each hypothesis-only training setup
pretrain_task_lists_ho = {
    "a9": "nli-a-ho",
    "b9": "nli-b-ho",
    "c9": "nli-c-ho",
    "d9": "nli-d-ho",
    "f9": "nli-f-ho",
    "m9": "mnli-ho",
    "g9": "mnli-government-ho",
    "ma": "mnli-ho",
    "x9": '\\"snli-ho,mnli-ho,adversarial_nli-ho\\"',
    "xa": '\\"snli-ho,mnli-ho,adversarial_nli-ho\\"',
}

nli_evaluation_settings_ho = {
    "a9": '\\"mnli-ho,nli-a-ho\\"',  # Intermediate training on our new data.
    "b9": '\\"mnli-ho,nli-b-ho\\"',  # Intermediate training on our new data.
    "c9": '\\"mnli-ho,nli-c-ho\\"',  # Intermediate training on our new data.
    "d9": '\\"mnli-ho,nli-d-ho\\"',  # Intermediate training on our new data.
    "f9": '\\"mnli-two-ho,nli-f-ho\\"',  # Intermediate training on our new data.
    "m9": '\\"mnli-ho,mnli-two-ho\\"',  # Intermediate training on a sample from MNLI.
    "g9": '\\"mnli-ho,mnli-government-ho\\"',  # Intermediate training on a single-genre sample from MNLI.
    "ma": '\\"mnli-ho,mnli-two-ho\\"',  # Intermediate training on MNLI.
    "x9": '\\"snli-ho,mnli-ho,adversarial_nli-ho\\"',  # Intermediate training on a sample from adversarial_nli (following standard practice by adding SNLI/MNLI for training).
    "xa": '\\"snli-ho,mnli-ho,adversarial_nli-ho\\"',  # Intermediate training on adversarial_nli (following standard practice by adding SNLI/MNLI for training).
}

# Set up hypothesis-only experiments
for config in ["hyp"]:
    do_pretrain_override = "1"
    do_target_task_training_override = "0"
    do_full_eval_override = "1"
    load_target_train_checkpoint_override = ""

    script = (
        "nli_data_evaluation_internal_local.conf"
    )  # No need to save checkpoints - these runs are fairly fast

    for model in ["roberta", "xlnet"]:
        model_override = models[model]

        for pretrain in ["a9", "b9", "c9", "d9", "f9", "m9", "g9", "ma", "x9", "xa"]:
            # No need for an intermediate training job in the no-intermediate-task setting

            pretrain_override = pretrain_task_lists_ho[pretrain]
            data_fraction_override = pretrain_data_fractions[pretrain]
            target_override = nli_evaluation_settings_ho[pretrain]

            for restart in range(3):
                runname = f"{BASENAME}-{config}-{model}-{pretrain}-{restart}"
                command = (
                    f"export RUNNAME={runname}; "
                    f'gcp/kubernetes/run_batch.sh $RUNNAME "python $JIANT_PATH/main.py '
                    f"--config_file $JIANT_PATH/jiant/config/{script} "
                    f"--overrides 'exp_name = {BASENAME}-{model}, run_name = $RUNNAME, "
                    f"pretrain_tasks = {pretrain_override}, target_tasks = {target_override}, "
                    f"do_pretrain = {do_pretrain_override}, input_module = {model_override}, "
                    f"pretrain_data_fraction = {data_fraction_override}, {load_target_train_checkpoint_override}"
                    f"do_target_task_training = {do_target_task_training_override}, "
                    f"do_full_eval = {do_full_eval_override}'\""
                )
                print(command)
        print("")
    print("")
