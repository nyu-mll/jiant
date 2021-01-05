import os
import torch
import torch.nn as nn

import jiant.utils.python.io as py_io
import jiant.utils.torch_utils as torch_utils
import jiant.proj.main.runner as runner
from jiant.proj.main.modeling.primary import wrap_jiant_forward
from jiant.ext.adapterfusion.adapter_bert import get_fusion_regularization_loss


class AdapterFusionRunner(runner.JiantRunner):
    def run_train_step(self, train_dataloader_dict: dict, train_state: runner.TrainState):
        self.jiant_model.train()
        task_name, task = self.jiant_task_container.task_sampler.pop()
        task_specific_config = self.jiant_task_container.task_specific_configs[task_name]

        loss_val = 0
        for i in range(task_specific_config.gradient_accumulation_steps):
            batch, batch_metadata = train_dataloader_dict[task_name].pop()
            batch = batch.to(self.device)
            model_output = wrap_jiant_forward(
                jiant_model=self.jiant_model, batch=batch, task=task, compute_loss=True,
            )

            loss = self.complex_backpropagate(
                loss=model_output.loss,
                gradient_accumulation_steps=task_specific_config.gradient_accumulation_steps,
            )
            loss_val += loss.item()

        # Apply fusion regularization only once per update, not per grad-accum step
        if (
            hasattr(self.model.encoder.config, "adapter_fusion")
            and self.model.encoder.config.adapter_fusion["regularization"]
        ):
            self.complex_backpropagate(
                loss=get_fusion_regularization_loss(self.model.encoder),
                gradient_accumulation_steps=task_specific_config.gradient_accumulation_steps,
            )

        self.optimizer_scheduler.step()
        self.optimizer_scheduler.optimizer.zero_grad()

        train_state.step(task_name=task_name)
        self.log_writer.write_entry(
            "loss_train",
            {
                "task": task_name,
                "task_step": train_state.task_steps[task_name],
                "global_step": train_state.global_steps,
                "loss_val": loss_val / task_specific_config.gradient_accumulation_steps,
            },
        )


def save_model_with_metadata(model: nn.Module, metadata: dict, output_dir: str, adapter_tuning_mode: str,
                             file_name="model"):
    save_model(
        model=model,
        output_dir=output_dir,
        adapter_tuning_mode=adapter_tuning_mode,
        file_name=file_name,
    )
    py_io.write_json(metadata, os.path.join(output_dir, f"{file_name}.metadata.json"))


def save_model(model: nn.Module, output_dir: str, adapter_tuning_mode: str, file_name="model"):
    raw_state_dict = torch_utils.get_model_for_saving(model).state_dict()
    save_directory = os.path.join(output_dir, file_name)
    os.makedirs(save_directory, exist_ok=True)
    if adapter_tuning_mode == "single":
        model.encoder.save_all_adapters(save_directory=save_directory)
    elif adapter_tuning_mode == "fusion":
        model.encoder.save_all_adapter_fusions(output_dir)
    else:
        raise KeyError(adapter_tuning_mode)
    state_dict = {
        n: p
        for n, p in raw_state_dict.items()
        if "encoder." not in n  # <-- this is a hack. Need better way to exclude encoder
    }
    torch.save(
        state_dict,
        os.path.join(output_dir, f"{file_name}.p"),
    )
