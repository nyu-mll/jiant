import json
import os
from shared_settings import batch_size_to_accumulation, batch_size_limit_to_gpus
from collect_trials import collect_trials

# TODO: clean up, move these decision-related variables to shared_settings
RESULT_DIR = ""
DATA_DIR = ""
RANDOM_SEEDS = [432, 5287, 98235, 8915, 2894]

metadata_file = os.path.join(os.path.dirname(__file__), "task_metadata.json")
with open(metadata_file, "r") as f:
    task_metadata = json.loads(f.read())


def preprocess_tasks(input_module):
    outputs = [
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(DATA_DIR, "edges/spr1/*.json")}" sbatch ~/cpu.sbatch',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(DATA_DIR, "edges/spr2/*.json")}" sbatch ~/cpu.sbatch',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(DATA_DIR, "edges/dpr/*.json")}" sbatch ~/cpu.sbatch',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(DATA_DIR, "edges/dep_ewt/*.json")}" sbatch ~/cpu.sbatch',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(DATA_DIR, "edges/ontonotes/const/pos/*.json")}" sbatch ~/cpu.sbatch',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(DATA_DIR, "edges/ontonotes/const/nonterminal/*.json")}" sbatch ~/cpu.sbatch',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(DATA_DIR, "edges/ontonotes/srl/*.json")}" sbatch ~/cpu.sbatch',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(DATA_DIR, "edges/ontonotes/ner/*.json")}" sbatch ~/cpu.sbatch',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(DATA_DIR, "edges/ontonotes/coref/*.json")}" sbatch ~/cpu.sbatch',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(DATA_DIR, "edges/semeval/*.json")}" sbatch ~/cpu.sbatch',
        f'PROG="scripts/ccg/align_tags_to_bert" ARGS="-t {input_module} -d {os.path.join(DATA_DIR, "ccg")}" sbatch ~/cpu.sbatch',
    ]
    return outputs


def run_exp_init(input_module):
    outputs = []

    task_names = list(set([task["task_name"] for task in task_metadata.values()]))
    exp_names = [f"batch_size_{input_module}"] + [
        f"exp_round{rid}_seed{seed}" for rid, seed in enumerate(RANDOM_SEEDS)
    ]
    for exp_name in exp_names:
        target_tasks = ",".join(task_names)
        override = f'exp_name={exp_name}, run_name=preprocess, target_tasks=\\"{target_tasks}\\"'
        outputs.append(
            f'JIANT_CONF="jiant/config/taskmaster/clean_roberta.conf" '
            f'JIANT_OVERRIDES="{override}" sbatch ~/jp40.sbatch'
        )
    return outputs


def run_batch_size_check(input_module):
    outputs = []
    for batch_size in [32, 16, 8, 4, 2, 1]:
        task_names = list(set([task["task_name"] for task in task_metadata.values()]))
        for task_name in task_names:
            val_interval = task_metadata[task_name]["training_size"] // batch_size
            override = (
                f"exp_name=batch_size_{input_module}, run_name={task_name}_{batch_size}, "
                f"do_pretrain=1, pretrain_tasks={task_name}, "
                f"input_module={input_module}, batch_size={batch_size}, "
                f"max_epochs=1, val_interval={val_interval}, "
                f"delete_checkpoints_when_done=1"
            )
            outputs.append(
                f'JIANT_CONF="jiant/config/taskmaster/clean_roberta.conf" '
                f'JIANT_OVERRIDES="{override}" sbatch ~/jp40.sbatch'
            )
    return outputs


def update_batch_size_check(input_module):
    task_batch_size_limit = {}
    for batch_size in [32, 16, 8, 4, 2, 1]:
        task_names = list(set([task["task_name"] for task in task_metadata.values()]))
        for task_name in task_names:
            exp_name = f"batch_size_{input_module}"
            run_name = f"{task_name}_{batch_size}"
            results_tsv = os.path.join(RESULT_DIR, exp_name, "results.tsv")
            if os.path.exists(results_tsv):
                with open(results_tsv, "r") as f:
                    results = dict([line.split("\t") for line in f.read().split("\n") if line])
                if run_name in results:
                    if (
                        task_name not in task_batch_size_limit
                        or batch_size > task_batch_size_limit[task_name]
                    ):
                        task_batch_size_limit[task_name] = batch_size
    for full_task_name, task in task_metadata.items():
        if task["task_name"] in task_batch_size_limit:
            batch_size_limit = task_batch_size_limit[task["task_name"]]
            task[f'{input_module.split("-")[0]}_batch_size_limit'] = batch_size_limit

    with open(metadata_file, "w") as f:
        f.write(json.dumps(task_metadata))


def run_main_optuna_trials(input_module):
    outputs = []
    for full_task_name, task in task_metadata.items():
        print(input_module, full_task_name)
        df_grouped = collect_trials(full_task_name, input_module)[1]
        batch_size_limit = task[f'{input_module.split("-")[0]}_batch_size_limit']
        gpu_available, sbatch = batch_size_limit_to_gpus(batch_size_limit, jiant=False)

        if full_task_name.endswith("20k"):
            training_size = 20000
        elif full_task_name.endswith("5k"):
            training_size = 5000
        else:
            training_size = task["training_size"]

        if training_size >= 100000:
            parallel = 4
        elif training_size >= 20000:
            parallel = 2
        else:
            parallel = 1

        if df_grouped is None:
            previous_trials = 0
            print("previous trials not found")
        else:
            previous_trials = sum(df_grouped["count"])
            print(f"previous trials found: {previous_trials} / 10")

        remaining_trials = 10 - previous_trials
        if remaining_trials <= 0:
            continue
        for i in range(parallel):
            num_trials = (remaining_trials // parallel) + (i < (remaining_trials % parallel))
            if num_trials == 0:
                continue
            outputs.append(
                f'PROG="scripts/taskmaster/optuna_hp_search/run_trials" ARGS="'
                f"--study-name {full_task_name} --gpu-available {gpu_available} "
                f'--n-trials {num_trials} --input-module {input_module}" sbatch ~/{sbatch}'
            )
    return outputs


def run_additional_optuna_trials(input_module):
    outputs = []
    for full_task_name, task in task_metadata.items():
        df_grouped = collect_trials(full_task_name, input_module)[1]
        if df_grouped is None or sum(df_grouped["count"]) < 10:
            print(f"{full_task_name} has not finished main optuna run.")
            continue
        batch_size_limit = task[f'{input_module.split("-")[0]}_batch_size_limit']
        gpu_available, sbatch = batch_size_limit_to_gpus(batch_size_limit, jiant=False)

        if df_grouped["count"][0] < 3:
            for rank in range(3):
                num_trials = max(0, 3 - df_grouped["count"][rank])
                if num_trials == 0:
                    continue
                outputs.append(
                    f'PROG="scripts/taskmaster/optuna_hp_search/run_trials" ARGS="'
                    f"--study-name {full_task_name} --gpu-available {gpu_available} "
                    f'--max-epochs {int(df_grouped["max_epochs"][rank])} --lr {float(df_grouped["lr"][rank])} --batch-size {int(df_grouped["batch_size"][rank])} '
                    f'--n-trials {num_trials} --input-module {input_module}" sbatch ~/{sbatch}'
                )
        else:
            task_metadata[full_task_name][f'{input_module.split("-")[0]}_hp'] = {
                "max_epochs": int(df_grouped["max_epochs"][0]),
                "lr": float(df_grouped["lr"][0]),
                "batch_size": int(df_grouped["batch_size"][0]),
            }
    with open(metadata_file, "w") as f:
        f.write(json.dumps(task_metadata))

    return outputs


def run_pretrain(
    input_module,
    include_mlm=True,
    include_single_task=True,
    include_full_size=True,
    include_20k_size=True,
):
    outputs = []
    checkponts = {}

    for full_task_name, task in task_metadata.items():
        batch_size_limit = task[f'{input_module.split("-")[0]}_batch_size_limit']
        gpu_available, sbatch = batch_size_limit_to_gpus(batch_size_limit, jiant=True)
        hp = task[f'{input_module.split("-")[0]}_hp']
        real_batch_size, accumulation_steps = batch_size_to_accumulation(
            batch_size_limit, hp["batch_size"], gpu_available
        )
        training_size = task["training_size"]
        if full_task_name.endswith("-5k"):
            data_fraction = 5000 / training_size
            training_size = 5000
        elif full_task_name.endswith("-20k"):
            data_fraction = 20000 / training_size
            training_size = 20000
        else:
            data_fraction = 1.0
        val_interval = max(training_size // hp["batch_size"], 5000)
        if (
            "I" not in task["role"]
            or (not include_20k_size and "20k" in full_task_name)
            or (not include_full_size and training_size > 20000)
        ):
            continue

        if include_single_task:
            run_name = f"interm_{full_task_name}"
            checkponts[run_name] = {}
            for rid, seed in enumerate(RANDOM_SEEDS):
                exp_name = f"exp_round{rid}_seed{seed}"
                override = (
                    f"exp_name={exp_name}, run_name={run_name}, random_seed={seed}, load_model=1, "
                    f'do_pretrain=1, pretrain_tasks={task["task_name"]}, input_module={input_module}, '
                    f'max_epochs={hp["max_epochs"]}, lr={hp["lr"]}, '
                    f"batch_size={real_batch_size}, accumulation_steps={accumulation_steps}, "
                    f"val_interval={val_interval}, pretrain_data_fraction={data_fraction}"
                )
                outputs.append(
                    f'JIANT_CONF="jiant/config/taskmaster/clean_roberta.conf" '
                    f'JIANT_OVERRIDES="{override}" sbatch ~/{sbatch}.sbatch'
                )
                checkponts[run_name][exp_name] = os.path.join(
                    RESULT_DIR, exp_name, run_name, "model_*.best.th"
                )

        if include_mlm:
            run_name = f"interm_{full_task_name}_mlm"
            checkponts[run_name] = {}
            for rid, seed in enumerate(RANDOM_SEEDS):
                exp_name = f"exp_round{rid}_seed{seed}"
                if input_module.split("-")[0] == "roberta":
                    mlm_val_interval = val_interval * 2
                    mlm_pretrain_tasks = '\\"{task["task_name"]},wikipedia_corpus_mlm\\"'
                elif input_module.split("-")[0] == "albert":
                    mlm_val_interval = val_interval * 3
                    mlm_pretrain_tasks = (
                        '\\"{task["task_name"]},wikipedia_corpus_mlm,wikipedia_corpus_sop\\"'
                    )
                override = (
                    f"exp_name={exp_name}, run_name={run_name}, random_seed={seed}, load_model=1, "
                    f"do_pretrain=1, pretrain_tasks={mlm_pretrain_tasks}, "
                    f'weighting_method=examples_proportional_mixingK=16384, early_stopping={task["task_name"]}'
                    f'input_module={input_module}, max_epochs={hp["max_epochs"]}, lr={hp["lr"]}, '
                    f"batch_size={real_batch_size}, accumulation_steps={accumulation_steps}, "
                    f"val_interval={mlm_val_interval}, pretrain_data_fraction={data_fraction}"
                )
                outputs.append(
                    f'JIANT_CONF="jiant/config/taskmaster/clean_roberta.conf" '
                    f'JIANT_OVERRIDES="{override}" sbatch ~/{sbatch}.sbatch'
                )
                checkponts[run_name][exp_name] = os.path.join(
                    RESULT_DIR, exp_name, run_name, "model_*.best.th"
                )

    for rid, seed in enumerate(RANDOM_SEEDS):
        exp_name = f"exp_round{rid}_seed{seed}"
        checkponts["baseline"][exp_name] = "none"

    return outputs, checkponts


def run_target_train(
    input_module,
    pretrain_checkponts,
    include_target=True,
    include_full_probing=True,
    include_5k_proibng=True,
):
    outputs = []

    for full_task_name, task in task_metadata.items():
        for pretrain_run_name in pretrain_checkponts:
            batch_size_limit = task[f'{input_module.split("-")[0]}_batch_size_limit']
            gpu_available, sbatch = batch_size_limit_to_gpus(batch_size_limit, jiant=True)
            hp = task[f'{input_module.split("-")[0]}_hp']
            real_batch_size, accumulation_steps = batch_size_to_accumulation(
                batch_size_limit, hp["batch_size"], gpu_available
            )
            training_size = task["training_size"]
            if full_task_name.endswith("-5k"):
                data_fraction = 5000 / training_size
                training_size = 5000
            elif full_task_name.endswith("-20k"):
                data_fraction = 20000 / training_size
                training_size = 20000
            else:
                data_fraction = 1.0
            val_interval = max(training_size // hp["batch_size"], 5000)
            if (include_target and "T" in task["role"]) or (
                (include_full_probing and "P" in task["role"] and "5k" not in full_task_name)
                or (include_5k_proibng and "P" in task["role"] and training_size <= 5000)
            ):
                pass
            else:
                continue

            run_name = f"{full_task_name}_from_{pretrain_run_name}"
            for rid, seed in enumerate(RANDOM_SEEDS):
                exp_name = f"exp_round{rid}_seed{seed}"
                override = (
                    f"exp_name={exp_name}, run_name={run_name}, random_seed={seed}, load_model=1, "
                    f'do_target_task_training=1, target_tasks={task["task_name"]}, input_module={input_module}, '
                    f'max_epochs={hp["max_epochs"]}, lr={hp["lr"]}, '
                    f"batch_size={real_batch_size}, accumulation_steps={accumulation_steps}, "
                    f"val_interval={val_interval}, target_train_data_fraction={data_fraction}"
                    f"load_target_train_checkpoint={pretrain_checkponts[pretrain_run_name][exp_name]}"
                )
                outputs.append(
                    f'JIANT_CONF="jiant/config/taskmaster/clean_roberta.conf" '
                    f'JIANT_OVERRIDES="{override}" sbatch ~/{sbatch}.sbatch'
                )

    return outputs


def write_script_file(script_name, outputs):
    with open(script_name, "w") as f:
        for line in outputs:
            f.write(line + "\n")
