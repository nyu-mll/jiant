import json
import os
from shared_settings import batch_size_to_accumulation, batch_size_limit_to_gpus
from collect_trials import collect_trails

# TODO: clean up, move these decision-related variables to shared_settings
RESULT_DIR = ""
DATA_DIR = ""
RANDOM_SEEDS = [432, 5287, 98235, 8915, 2894]

with open("task_metadata.json", "r") as f:
    task_metadata = json.loads(f.read())


def preprocess_tasks(input_module):
    outputs = [
        f'srun --mem=10000 --nodes=1 python probing/retokenize_edge_data.py -t {input_module} {os.path.join(DATA_DIR, "edges/spr1/*.json")}',
        f'srun --mem=10000 --nodes=1 python probing/retokenize_edge_data.py -t {input_module} {os.path.join(DATA_DIR, "edges/spr2/*.json")}',
        f'srun --mem=10000 --nodes=1 python probing/retokenize_edge_data.py -t {input_module} {os.path.join(DATA_DIR, "edges/dpr/*.json")}',
        f'srun --mem=10000 --nodes=1 python probing/retokenize_edge_data.py -t {input_module} {os.path.join(DATA_DIR, "edges/dep_ewt/*.json")}',
        f'srun --mem=10000 --nodes=1 python probing/retokenize_edge_data.py -t {input_module} {os.path.join(DATA_DIR, "edges/ontonotes/const/pos/*.json")}',
        f'srun --mem=10000 --nodes=1 python probing/retokenize_edge_data.py -t {input_module} {os.path.join(DATA_DIR, "edges/ontonotes/const/nonterminal/*.json")}',
        f'srun --mem=10000 --nodes=1 python probing/retokenize_edge_data.py -t {input_module} {os.path.join(DATA_DIR, "edges/ontonotes/srl/*.json")}',
        f'srun --mem=10000 --nodes=1 python probing/retokenize_edge_data.py -t {input_module} {os.path.join(DATA_DIR, "edges/ontonotes/ner/*.json")}',
        f'srun --mem=10000 --nodes=1 python probing/retokenize_edge_data.py -t {input_module} {os.path.join(DATA_DIR, "edges/ontonotes/coref/*.json")}',
        f'srun --mem=10000 --nodes=1 python probing/retokenize_edge_data.py -t {input_module} {os.path.join(DATA_DIR, "edges/semeval/*.json")}',
        f'srun --mem=10000 --nodes=1 python scripts/ccg/align_tags_to_bert.py -t {input_module} -d {os.path.join(DATA_DIR, "ccg")}',
    ]
    return outputs


def create_experiment(input_module):
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
            f"JIANT_OVERRIDES={override} sbatch ~/jp40.sbatch"
        )
    return outputs


def run_batch_size_check(input_module):
    outputs = []
    for batch_size in [32, 16, 8, 4, 2]:
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
                f"JIANT_OVERRIDES={override} sbatch ~/jp40.sbatch"
            )
    return outputs


def update_batch_size_check(input_module):
    task_batch_size_limit = {}
    for batch_size in [32, 16, 8, 4, 2]:
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

    with open("task_metadata.json", "w") as f:
        f.write(json.dumps(task_metadata))


def run_main_optuna_trails(input_module):
    outputs = []
    for full_task_name, task in task_metadata.items():
        df_grouped = collect_trails(full_task_name, input_module)[1]
        batch_size_limit = task[f'{input_module.split("-")[0]}_batch_size_limit']
        gpu_available, sbatch = batch_size_limit_to_gpus(batch_size_limit)

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

        total_num_trails = 10 - sum(df_grouped["count"])
        for i in range(parallel):
            num_trails = (total_num_trails // parallel) + (i < (total_num_trails % parallel))
            if num_trails == 0:
                continue
            outputs.append(
                f'PROG="scripts/taskmaster/optuna_hp_search/run_trials" ARGS="'
                f"--study-name {full_task_name} --gpu-available {gpu_available} "
                f'--num-trails {num_trails} --input_module {input_module}" sbatch ~/{sbatch}'
            )
    return outputs


def run_additional_optuna_trails(input_module):
    outputs = []
    for full_task_name, task in task_metadata.items():
        df_grouped = collect_trails(full_task_name, input_module)[1]
        batch_size_limit = task[f'{input_module.split("-")[0]}_batch_size_limit']
        gpu_available, sbatch = batch_size_limit_to_gpus(batch_size_limit)

        all_done = True
        for rank in range(3):
            num_trails = max(0, 3 - df_grouped["count"][rank])
            if num_trails == 0:
                continue
            all_done = False
            outputs.append(
                f'PROG="scripts/taskmaster/optuna_hp_search/run_trials" ARGS="'
                f"--study-name {full_task_name} --gpu-available {gpu_available} "
                f'--max-epochs {df_grouped["max_epochs"][rank]} --lr {df_grouped["lr"][rank]} --batch-size {df_grouped["batch_size"][rank]} '
                f'--num-trails {num_trails} --input_module {input_module}" sbatch ~/{sbatch}'
            )
        if all_done:
            task_metadata[full_task_name][f'{input_module.split("-")[0]}_hp'] = {
                "max_epochs": df_grouped["max_epochs"][0],
                "lr": df_grouped["lr"][0],
                "batch_size": df_grouped["batch_size"][0],
            }
    with open("task_metadata.json", "w") as f:
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
        gpu_available, sbatch = batch_size_limit_to_gpus(batch_size_limit)
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
                    f"JIANT_OVERRIDES={override} sbatch ~/{sbatch}.sbatch"
                )
                checkponts[run_name][exp_name] = os.path.join(
                    RESULT_DIR, exp_name, run_name, "model_*.best.th"
                )

        if include_mlm:
            run_name = f"interm_{full_task_name}_mlm"
            checkponts[run_name] = {}
            for rid, seed in enumerate(RANDOM_SEEDS):
                exp_name = f"exp_round{rid}_seed{seed}"
                override = (
                    f"exp_name={exp_name}, run_name={run_name}, random_seed={seed}, load_model=1, "
                    f'do_pretrain=1, pretrain_tasks=\\"{task["task_name"]},wikipedia_corpus_mlm\\", '
                    f'weighting_method=examples_proportional_mixingK=16384, early_stopping={task["task_name"]}'
                    f'input_module={input_module}, max_epochs={hp["max_epochs"]}, lr={hp["lr"]}, '
                    f"batch_size={real_batch_size}, accumulation_steps={accumulation_steps}, "
                    f"val_interval={val_interval}, pretrain_data_fraction={data_fraction}"
                )
                # TODO: double check how to decide val_interval
                outputs.append(
                    f'JIANT_CONF="jiant/config/taskmaster/clean_roberta.conf" '
                    f"JIANT_OVERRIDES={override} sbatch ~/{sbatch}.sbatch"
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
            gpu_available, sbatch = batch_size_limit_to_gpus(batch_size_limit)
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
                    f"JIANT_OVERRIDES={override} sbatch ~/{sbatch}.sbatch"
                )

    return outputs


def write_script_file(script_name, outputs):
    with open(script_name, "w") as f:
        for line in outputs:
            f.write(line + "\n")
