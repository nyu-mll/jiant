import os
import torch

import jiant.tasks as tasks
import jiant.utils.zconf as zconf
import jiant.proj.main.modeling.model_setup as model_setup
from jiant.proj.main.modeling.primary import JiantModel, wrap_jiant_forward
import jiant.proj.main.modeling.taskmodels as taskmodels
import jiant.proj.main.modeling.heads as heads
import jiant.shared.model_setup as shared_model_setup
from jiant.shared.model_setup import ModelArchitectures
import jiant.shared.caching as caching
import jiant.utils.torch_utils as torch_utils


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    model_config_path = zconf.attr(type=str)
    model_tokenizer_path = zconf.attr(type=str)
    task_config_base_path = zconf.attr(type=str)
    task_cache_base_path = zconf.attr(type=str)


def main(args: RunConfiguration):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Shared model components setup === #
    model_type = "roberta-base"
    model_arch = ModelArchitectures.from_model_type(model_type=model_type)
    transformers_class_spec = model_setup.TRANSFORMERS_CLASS_SPEC_DICT[model_arch]
    ancestor_model = model_setup.get_ancestor_model(
        transformers_class_spec=transformers_class_spec,
        model_config_path=args.model_config_path,
    )
    encoder = model_setup.get_encoder(
        model_arch=model_arch,
        ancestor_model=ancestor_model,
    )
    tokenizer = shared_model_setup.get_tokenizer(
        model_type=model_type,
        tokenizer_path=args.model_tokenizer_path,
    )

    # === Taskmodels setup === #
    task_dict = {
        "mnli": tasks.create_task_from_config_path(os.path.join(
            args.task_config_base_path, "mnli.json",
        )),
        "qnli": tasks.create_task_from_config_path(os.path.join(
            args.task_config_base_path, "qnli.json",
        )),
        "rte": tasks.create_task_from_config_path(os.path.join(
            args.task_config_base_path, "qnli.json",
        ))
    }
    taskmodels_dict = {
        "nli": taskmodels.ClassificationModel(
            encoder=encoder,
            classification_head=heads.ClassificationHead(
                hidden_size=encoder.config.hidden_size,
                hidden_dropout_prob=encoder.config.hidden_dropout_prob,
                num_labels=len(task_dict["mnli"].LABELS),
            ),
        ),
        "rte": taskmodels.ClassificationModel(
            encoder=encoder,
            classification_head=heads.ClassificationHead(
                hidden_size=encoder.config.hidden_size,
                hidden_dropout_prob=encoder.config.hidden_dropout_prob,
                num_labels=len(task_dict["rte"].LABELS),
            ),
        ),
    }
    task_to_taskmodel_map = {
        "mnli": "nli",
        "qnli": "nli",
        "rte": "rte",
    }

    # === Final === #
    jiant_model = JiantModel(
        task_dict=task_dict,
        encoder=encoder,
        taskmodels_dict=taskmodels_dict,
        task_to_taskmodel_map=task_to_taskmodel_map,
        tokenizer=tokenizer,
    )
    jiant_model = jiant_model.to(device)

    # === Run === #
    task_dataloader_dict = {}
    for task_name, task in task_dict.items():
        train_cache = caching.ChunkedFilesDataCache(
            cache_fol_path=os.path.join(args.task_cache_base_path, task_name, "train"),
        )
        train_dataset = train_cache.get_iterable_dataset(buffer_size=10000, shuffle=True)
        train_dataloader = torch_utils.DataLoaderWithLength(
            dataset=train_dataset,
            batch_size=4,
            collate_fn=task.collate_fn,
        )
        task_dataloader_dict[task_name] = train_dataloader

    for task_name, task in task_dict.items():
        batch, batch_metadata = next(iter(task_dataloader_dict[task_name]))
        batch = batch.to(device)
        with torch.no_grad():
            model_output = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=True,
            )
        print(task_name)
        print(model_output)
        print()


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
