from dataclasses import dataclass
from typing import Dict
import torch.nn as nn


import jiant.proj.main.runner as jiant_runner
import jiant.proj.main.components.task_sampler as jiant_task_sampler
from jiant.shared.runner import (
    save_model_with_metadata,
    compare_steps_max_steps,
)
from jiant.utils.python.datastructures import ExtendedDataClassMixin
from jiant.utils.python.functional import always_false
from jiant.utils.torch_utils import copy_state_dict, CPU_DEVICE, get_model_for_saving
from jiant.utils.zlog import BaseZLogger, PRINT_LOGGER
from jiant.shared.metarunner import AbstractMetarunner


@dataclass
class ValState(ExtendedDataClassMixin):
    score: float
    metrics: Dict
    train_state: jiant_runner.TrainState

    def new(self):
        # noinspection PyArgumentList
        return self.__class__(
            score=self.score, metrics=self.metrics, train_state=self.train_state.new(),
        )

    def to_dict(self):
        return {
            "score": float(self.score),
            "metrics": self.metrics,
            "train_state": self.train_state.to_dict(),
        }


def get_should_early_stop_func(eval_every_steps: int, no_improvements_for_n_evals: int):
    assert eval_every_steps != 0
    if no_improvements_for_n_evals == 0:
        return always_false
    else:
        return lambda metarunner: (
            metarunner.train_state.global_steps is not None
            and metarunner.best_val_state is not None
            and (
                metarunner.best_val_state.train_state.global_steps
                - metarunner.train_state.global_steps
            )
            / eval_every_steps
            > no_improvements_for_n_evals
        )


class JiantMetarunner(AbstractMetarunner):
    def __init__(
        self,
        runner: jiant_runner.JiantRunner,
        save_every_steps,
        eval_every_steps,
        save_checkpoint_every_steps,
        no_improvements_for_n_evals,
        checkpoint_saver,
        output_dir,
        verbose: bool = True,
        save_best_model: bool = True,
        load_best_model: bool = True,
        save_last_model: bool = True,
        log_writer: BaseZLogger = PRINT_LOGGER,
    ):
        self.runner = runner
        self.save_every_steps = save_every_steps
        self.eval_every_steps = eval_every_steps
        self.save_checkpoint_every_steps = save_checkpoint_every_steps
        self.no_improvements_for_n_evals = no_improvements_for_n_evals
        self.checkpoint_saver = checkpoint_saver
        self.output_dir = output_dir
        self.verbose = verbose
        self.save_best_model = save_best_model
        self.load_best_model = load_best_model
        self.save_last_model = save_last_model
        self.log_writer = log_writer

        self.best_val_state = None
        self.best_state_dict = None
        self.val_state_history = []
        self.train_state = None
        self.full_break = False
        self.single_use_check = False
        self.num_evals_since_improvement = 0

        self.model = self.runner.model
        self.device = self.runner.device
        self.global_train_config = self.runner.jiant_task_container.global_train_config

    def begin_training(self):
        assert not self.single_use_check
        self.single_use_check = True

    def yield_train_step(self):
        if self.train_state is None:
            # Fresh run
            train_iterator = self.runner.run_train_context(verbose=self.verbose)
        else:
            train_iterator = self.runner.resume_train_context(
                train_state=self.train_state, verbose=self.verbose,
            )
        for train_state in train_iterator:
            self.train_state = train_state
            self.inject_at_step()
            yield

    def should_save_model(self) -> bool:
        if self.save_every_steps == 0:
            return False
        return (self.train_state.global_steps + 1) % self.save_every_steps == 0

    def save_model(self):
        save_model_with_metadata(
            model_or_state_dict=self.model,
            output_dir=self.output_dir,
            file_name=f"model__{self.train_state.global_steps:09d}",
        )

    def save_last_model_with_metadata(self):
        save_model_with_metadata(
            model_or_state_dict=self.model,
            output_dir=self.output_dir,
            file_name="last_model",
            metadata={"train_state": self.train_state.to_dict()},
        )

    def save_best_model_with_metadata(self, val_metrics_dict):
        save_model_with_metadata(
            model_or_state_dict=self.best_state_dict,
            output_dir=self.output_dir,
            file_name="best_model",
            metadata={
                "val_state": self.best_val_state.to_dict(),
                "val_metrics": self.best_val_state.metrics,
            },
        )

    def should_save_checkpoint(self) -> bool:
        if self.save_checkpoint_every_steps == 0:
            return False
        return (self.train_state.global_steps + 1) % self.save_checkpoint_every_steps == 0

    def save_checkpoint(self):
        runner_state = self.runner.get_runner_state()
        metarunner_state = self.get_state()
        print("Saving State")
        self.checkpoint_saver.save(runner_state=runner_state, metarunner_state=metarunner_state)

    def should_eval_model(self) -> bool:
        if self.eval_every_steps == 0:
            return False
        return (self.train_state.global_steps + 1) % self.eval_every_steps == 0

    def eval_model(self):
        self.eval_save()

    def should_break_training(self) -> bool:
        if compare_steps_max_steps(
            step=self.train_state.global_steps, max_steps=self.global_train_config.max_steps
        ):
            return True

        if self.no_improvements_for_n_evals != 0:
            if self.num_evals_since_improvement >= self.no_improvements_for_n_evals:
                self.log_writer.write_entry(
                    "early_stopping",
                    {"message": "early_stopped", "train_state": self.train_state.to_dict()},
                )
                self.log_writer.flush()
                return True

        return False

    def done_training(self):
        if self.save_last_model:
            self.save_last_model_with_metadata()
        self.eval_save()
        if self.load_best_model and self.best_state_dict is not None:
            if self.verbose:
                print("Loading Best")
            copied_state_dict = copy_state_dict(
                state_dict=self.best_state_dict,
                target_device=None,  # Why was this required?
                # target_device=self.device,
            )
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(copied_state_dict)
            else:
                self.model.load_state_dict(copied_state_dict)

    def returned_result(self):
        return {
            "best_val_state": self.best_val_state,
            "val_state_history": self.val_state_history,
        }

    # ======================== #

    def inject_at_step(self):
        pass

    def get_state(self):
        return {
            "best_val_state": self.best_val_state,
            "best_state_dict": self.best_state_dict,
            "train_state": self.train_state,
        }

    def load_state(self, metarunner_state):
        self.best_val_state = metarunner_state["best_val_state"]
        self.best_state_dict = metarunner_state["best_state_dict"]
        self.train_state = metarunner_state["train_state"]

    def eval_save(self):
        self.num_evals_since_improvement += 1
        val_results_dict = self.runner.run_val(
            task_name_list=self.runner.jiant_task_container.task_run_config.train_val_task_list,
            use_subset=True,
        )
        aggregated_major = jiant_task_sampler.compute_aggregate_major_metrics_from_results_dict(
            metrics_aggregator=self.runner.jiant_task_container.metrics_aggregator,
            results_dict=val_results_dict,
        )
        val_metrics_dict = jiant_task_sampler.get_metrics_dict_from_results_dict(
            results_dict=val_results_dict,
        )
        val_state = ValState(
            score=float(aggregated_major),
            metrics=val_metrics_dict,
            train_state=self.train_state.new(),
        )
        self.log_writer.write_entry("train_val", val_state.to_dict())
        if self.best_val_state is None or val_state.score > self.best_val_state.score:
            self.best_val_state = val_state.new()
            self.log_writer.write_entry("train_val_best", self.best_val_state.to_dict())
            del self.best_state_dict
            self.best_state_dict = copy_state_dict(
                state_dict=get_model_for_saving(self.model).state_dict(), target_device=CPU_DEVICE,
            )
            if self.save_best_model:
                self.save_best_model_with_metadata(val_metrics_dict=val_metrics_dict)
            self.num_evals_since_improvement = 0
        self.log_writer.write_entry(
            "early_stopping",
            {
                "num_evals_since_improvement": self.num_evals_since_improvement,
                "train_state": self.train_state.to_dict(),
            },
        )
        self.log_writer.flush()
        self.val_state_history.append(val_state)
