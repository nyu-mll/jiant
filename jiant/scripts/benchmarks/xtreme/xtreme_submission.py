import os
import numpy as np
import torch

import zconf
import jiant.utils.python.io as py_io
from jiant.utils.display import maybe_tqdm

import jiant.shared.initialization as initialization
import jiant.shared.model_resolution as model_resolution
import jiant.proj.main.components.container_setup as container_setup
import jiant.proj.main.runner as jiant_runner
from jiant.proj.main.runscript import setup_runner
import jiant.tasks.lib.templates.squad_style.core as squad_lib
import jiant.tasks.lib.bucc2018 as bucc2018_lib
import jiant.tasks as tasks
import jiant.proj.main.components.evaluate as jiant_evaluate
from jiant.proj.main.modeling.primary import wrap_jiant_forward


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === RunConfig Parameters === #
    jiant_task_container_path = zconf.attr(type=str, default=None)

    # === Required Parameters === #
    supertask = zconf.attr(type=str, default=None)
    output_dir = zconf.attr(type=str, required=True)

    # === Optional Parameters === #
    skip_if_done = zconf.attr(action="store_true")
    bucc_val_metrics_path = zconf.attr(
        type=str,
        default=None,
        help="Path to val_metrics.json for bucc2018. Contains the optimal threshold,"
        " to be used for generating test predictions.",
    )

    # === Model parameters === #
    model_type = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(default=None, type=str)
    model_load_mode = zconf.attr(default="from_ptt", type=str)

    # === Nuisance Parameters === #
    # Required for quickly setting up runner
    # Remove/refactor with config refactor (issue #1176)
    learning_rate = zconf.attr(default=1e-5, type=float)
    adam_epsilon = zconf.attr(default=1e-8, type=float)
    max_grad_norm = zconf.attr(default=1.0, type=float)
    optimizer_type = zconf.attr(default="adam", type=str)

    # Specialized config
    no_cuda = zconf.attr(action="store_true")
    fp16 = zconf.attr(action="store_true")
    fp16_opt_level = zconf.attr(default="O1", type=str)
    local_rank = zconf.attr(default=-1, type=int)
    server_ip = zconf.attr(default="", type=str)
    server_port = zconf.attr(default="", type=str)
    force_overwrite = zconf.attr(action="store_true")
    seed = zconf.attr(type=int, default=-1)


def run_loop(args: RunConfiguration):
    quick_init_out = initialization.quick_init(args=args, verbose=True)
    with quick_init_out.log_writer.log_context():
        if args.jiant_task_container_path:
            jiant_task_container = container_setup.create_jiant_task_container(
                **py_io.read_json(args.jiant_task_container_path)
            )
        else:
            raise RuntimeError("Need `jiant_task_container_path` or individual config paths")
        runner = setup_runner(
            args=args,
            jiant_task_container=jiant_task_container,
            quick_init_out=quick_init_out,
            verbose=True,
        )
    supertask, output_dir = args.supertask, args.output_dir
    if supertask in ["xnli", "pawsx"]:
        generate_and_write_preds_for_classification(
            runner=runner,
            supertask=supertask,
            output_dir=output_dir,
            skip_if_done=args.skip_if_done,
        )
    elif supertask in ["udpos", "panx"]:
        generate_and_write_preds_for_tagging(
            runner=runner,
            supertask=supertask,
            output_dir=output_dir,
            skip_if_done=args.skip_if_done,
        )
    elif supertask in ["xquad", "mlqa"]:
        generate_and_write_preds_for_qa(
            runner=runner,
            supertask=supertask,
            output_dir=output_dir,
            phase="test",
            skip_if_done=args.skip_if_done,
        )
    elif supertask == "tydiqa":
        generate_and_write_preds_for_qa(
            runner=runner,
            supertask="tydiqa",
            output_dir=output_dir,
            phase="val",
            skip_if_done=args.skip_if_done,
        )
    elif supertask == "bucc2018":
        generate_and_write_preds_for_bucc2018(
            runner=runner,
            output_dir=output_dir,
            bucc_val_metrics_path=args.bucc_val_metrics_path,
            skip_if_done=args.skip_if_done,
        )
    elif supertask == "tatoeba":
        generate_and_write_preds_for_tatoeba(
            runner=runner, output_dir=output_dir, skip_if_done=args.skip_if_done,
        )
    else:
        raise KeyError(supertask)


def generate_and_write_preds_for_classification(
    runner: jiant_runner.JiantRunner, supertask: str, output_dir: str, skip_if_done: bool = False
):
    """Write test predictions for classification tasks in XTREME submission format"""
    preds_pickle_path = os.path.join(output_dir, f"{supertask}_test_preds.p")
    if skip_if_done and os.path.exists(preds_pickle_path):
        print(f"Skipping cause {preds_pickle_path} exists")
        return

    test_results_dict = runner.run_test(
        task_name_list=runner.jiant_task_container.task_run_config.test_task_list,
    )
    jiant_evaluate.write_preds(
        eval_results_dict=test_results_dict, path=preds_pickle_path,
    )
    preds_output_dir = os.path.join(output_dir, "preds", supertask)
    os.makedirs(preds_output_dir, exist_ok=True)
    for task_name, task_results in test_results_dict.items():
        task = runner.jiant_task_container.task_dict[task_name]
        assert isinstance(task, (tasks.XnliTask, tasks.PawsXTask))
        lang = task.language
        with open(os.path.join(preds_output_dir, f"test-{lang}.tsv"), "w") as f:
            for idx in task_results["preds"]:
                if supertask == "xnli":
                    pred_label = task.ID_TO_LABEL[idx]
                elif supertask == "pawsx":
                    pred_label = idx
                else:
                    raise RuntimeError()
                f.write(f"{pred_label}\n")
    print(f"Wrote {supertask} preds for {len(test_results_dict)} languages")


def generate_and_write_preds_for_tagging(
    runner: jiant_runner.JiantRunner, supertask: str, output_dir: str, skip_if_done: bool = False
):
    """Generate and write test predictions for tagging tasks in XTREME submission format"""
    preds_pickle_path = os.path.join(output_dir, f"{supertask}_test_preds.p")
    if skip_if_done and os.path.exists(preds_pickle_path):
        print(f"Skipping cause {preds_pickle_path} exists")
        return

    test_dataloader_dict = runner.get_test_dataloader_dict()
    preds_output_dir = os.path.join(output_dir, "preds", supertask)
    os.makedirs(preds_output_dir, exist_ok=True)
    preds_dict = {}
    for task_name in runner.jiant_task_container.task_run_config.test_task_list:
        task = runner.jiant_task_container.task_dict[task_name]
        assert isinstance(task, (tasks.UdposTask, tasks.PanxTask))
        preds_list = get_preds_for_single_tagging_task(
            task=task, test_dataloader=test_dataloader_dict[task_name], runner=runner,
        )
        preds_dict[task_name] = preds_list
        lang = task.language
        with open(os.path.join(preds_output_dir, f"test-{lang}.tsv"), "w") as f:
            for example_preds in preds_list:
                for word, label in example_preds:
                    f.write(f"{label}\n")
                f.write("\n")
    torch.save(preds_dict, preds_pickle_path)
    print(
        f"Wrote {supertask} preds for"
        f" {len(runner.jiant_task_container.task_run_config.test_task_list)} languages"
    )


def get_preds_for_single_tagging_task(
    task, test_dataloader, runner: jiant_runner.JiantRunner, verbose: str = True
):
    """Generate predictions for a single tagging task"""
    jiant_model, device = runner.model, runner.device
    jiant_model.eval()
    test_examples = task.get_test_examples()
    preds_list = []
    example_i = 0
    for step, (batch, batch_metadata) in enumerate(
        maybe_tqdm(test_dataloader, desc=f"Eval ({task.name}, Test)", verbose=verbose)
    ):
        batch = batch.to(device)

        with torch.no_grad():
            model_output = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=False,
            )
        batch_logits = model_output.logits.detach().cpu().numpy()
        label_mask_arr = batch.label_mask.cpu().bool().numpy()
        preds_arr = np.argmax(batch_logits, axis=-1)
        for i in range(len(batch)):
            # noinspection PyUnresolvedReferences
            labels = [task.ID_TO_LABEL[class_i] for class_i in preds_arr[i][label_mask_arr[i]]]
            if len(labels) == len(test_examples[example_i].tokens):
                this_preds = list(zip(test_examples[example_i].tokens, labels))
            elif len(labels) < len(test_examples[example_i].tokens):
                this_preds = list(zip(test_examples[example_i].tokens, labels))
                this_preds += [
                    (task.LABELS[-1], token)
                    for token in test_examples[example_i].tokens[len(labels) :]
                ]
            else:
                raise RuntimeError

            preds_list.append(this_preds)
            example_i += 1
    return preds_list


def generate_and_write_preds_for_qa(
    runner, supertask: str, output_dir: str, phase: str, skip_if_done: bool = False
):
    """Generate predictions (test) for QA tasks and write them in XTREME submission format"""
    preds_pickle_path = os.path.join(output_dir, f"{supertask}_test_preds.p")
    if skip_if_done and os.path.exists(preds_pickle_path):
        print(f"Skipping cause {preds_pickle_path} exists")
        return

    if phase == "val":
        task_name_list = runner.jiant_task_container.task_run_config.val_task_list
    elif phase == "test":
        task_name_list = runner.jiant_task_container.task_run_config.test_task_list
    else:
        raise KeyError(phase)
    task_name_list = [task_name for task_name in task_name_list if task_name.startswith(supertask)]
    if phase == "val":
        test_results_dict = runner.run_val(task_name_list=task_name_list)
    elif phase == "test":
        test_results_dict = {}
        test_dataloader_dict = runner.get_test_dataloader_dict()
        for task_name in task_name_list:
            test_results_dict[task_name] = jiant_runner.run_test(
                test_dataloader=test_dataloader_dict[task_name],
                jiant_model=runner.jiant_model,
                task=runner.jiant_task_container.task_dict[task_name],
                device=runner.device,
                local_rank=runner.rparams.local_rank,
                return_preds=False,
                verbose=True,
            )
    else:
        raise KeyError(phase)

    # Generate QA preds
    tokenizer = runner.model.tokenizer
    for task_name in task_name_list:
        task_results = test_results_dict[task_name]
        task = runner.jiant_task_container.task_dict[task_name]
        logits = task_results["accumulator"].get_accumulated()
        lang = get_qa_language(supertask=supertask, task=task)
        if phase == "val":
            cached = runner.get_val_dataloader_dict([task_name])[
                task_name
            ].dataset.chunked_file_data_cache.get_all()
        elif phase == "test":
            cached = runner.get_test_dataloader_dict()[
                task_name
            ].dataset.chunked_file_data_cache.get_all()
        else:
            raise KeyError(phase)
        data_rows = [row["data_row"] for row in cached]
        results, predictions = squad_lib.compute_predictions_logits_v3(
            data_rows=data_rows,
            logits=logits,
            n_best_size=task.n_best_size,
            max_answer_length=task.max_answer_length,
            do_lower_case=model_resolution.resolve_is_lower_case(tokenizer),
            version_2_with_negative=task.version_2_with_negative,
            null_score_diff_threshold=task.null_score_diff_threshold,
            skip_get_final_text=(lang == "zh"),
            tokenizer=tokenizer,
        )
        test_results_dict[task_name]["preds"] = predictions

    jiant_evaluate.write_preds(
        eval_results_dict=test_results_dict, path=preds_pickle_path,
    )
    preds_output_dir = os.path.join(output_dir, "preds", supertask)
    os.makedirs(preds_output_dir, exist_ok=True)
    for task_name, task_results in test_results_dict.items():
        task = runner.jiant_task_container.task_dict[task_name]
        lang = get_qa_language(supertask=supertask, task=task)
        py_io.write_json(task_results["preds"], os.path.join(preds_output_dir, f"test-{lang}.json"))
    print(f"Wrote {supertask} preds for {len(test_results_dict)} languages")


def get_qa_language(supertask: str, task):
    """Identify language from QA task"""
    if supertask == "mlqa":
        assert task.context_language == task.question_language
        lang = task.context_language
    elif supertask in ["xquad", "tydiqa"]:
        lang = task.language
    else:
        raise KeyError(supertask)
    return lang


def generate_and_write_preds_for_bucc2018(
    runner, output_dir: str, bucc_val_metrics_path: str, skip_if_done: bool = False
):
    """Generate predictions (test) for Bucc2018 and write them in XTREME submission format"""
    preds_pickle_path = os.path.join(output_dir, "bucc2018_test_preds.p")
    if skip_if_done and os.path.exists(preds_pickle_path):
        print(f"Skipping cause {preds_pickle_path} exists")
        return
    else:
        print(f"{preds_pickle_path} does not exist")
    if bucc_val_metrics_path is None:
        # Recompute thresholds:
        val_results_dict = runner.run_val(
            task_name_list=runner.jiant_task_container.task_run_config.val_task_list,
            return_preds=True,
        )
        jiant_evaluate.write_preds(
            eval_results_dict=val_results_dict,
            path=os.path.join(output_dir, "bucc2018_val_preds.p"),
        )
        thresholds_dict = {
            task_name: task_results["metrics"].minor["best-threshold"]
            for task_name, task_results in val_results_dict.items()
        }
    else:
        val_metrics = py_io.read_json(bucc_val_metrics_path)
        thresholds_dict = {
            task_name: val_metrics[task_name]["metrics"]["minor"]["best-threshold"]
            for task_name in runner.jiant_task_container.task_run_config.val_task_list
        }

    preds_output_dir = os.path.join(output_dir, "preds", "bucc2018")
    os.makedirs(preds_output_dir, exist_ok=True)
    test_results_dict = runner.run_test(
        task_name_list=runner.jiant_task_container.task_run_config.test_task_list,
    )
    jiant_evaluate.write_preds(
        eval_results_dict=test_results_dict, path=preds_pickle_path,
    )
    for task_name, task_results in test_results_dict.items():
        bitext = bucc2018_lib.bucc_extract(
            cand2score=task_results["preds"], th=thresholds_dict[task_name],
        )
        lang = runner.jiant_task_container.task_dict[task_name].language
        with open(os.path.join(preds_output_dir, f"test-{lang}.tsv"), "w") as f:
            for src, trg in bitext:
                f.write(f"{src}\t{trg}\n")
    print(f"Wrote Bucc2018 preds for {len(test_results_dict)} languages")


def generate_and_write_preds_for_tatoeba(runner, output_dir: str, skip_if_done: bool = False):
    """Generate predictions (val) for Tateoba and write them in XTREME submission format"""
    preds_pickle_path = os.path.join(output_dir, "tatoeba_val_preds.p")
    if skip_if_done and os.path.exists(preds_pickle_path):
        print(f"Skipping cause {preds_pickle_path} exists")
        return
    val_results_dict = runner.run_val(
        task_name_list=runner.jiant_task_container.task_run_config.val_task_list, return_preds=True,
    )
    jiant_evaluate.write_preds(
        eval_results_dict=val_results_dict, path=preds_pickle_path,
    )
    preds_output_dir = os.path.join(output_dir, "preds", "tatoeba")
    os.makedirs(preds_output_dir, exist_ok=True)
    for task_name, task_results in val_results_dict.items():
        lang = runner.jiant_task_container.task_dict[task_name].language
        with open(os.path.join(preds_output_dir, f"test-{lang}.tsv"), "w") as f:
            for idx in task_results["preds"]:
                f.write(f"{idx:d}\n")
    print(f"Wrote Tatoeba preds for {len(val_results_dict)} languages")


def main():
    run_loop(RunConfiguration.default_run_cli())


if __name__ == "__main__":
    main()
