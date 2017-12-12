"""
A :class:`~allennlp.training.MultiTaskTrainer.MultiTaskTrainer` is responsible
for training a :class:`~allennlp.models.model.Model`.

Typically you might create a configuration file specifying the model and
training parameters and then use :mod:`~allennlp.commands.train`
rather than instantiating a ``MultiTaskTrainer`` yourself.
"""

import os
import pdb # pylint: disable=unused-import
import math
import time
import copy
import random
import shutil
import logging
import itertools
from typing import Dict, Optional, List

import torch
import torch.optim.lr_scheduler
from torch.nn.utils.clip_grad import clip_grad_norm
import tqdm
from tensorboard import SummaryWriter

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from allennlp.nn.util import arrays_to_variables, device_mapping
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class MultiTaskTrainer:
    def __init__(self,
                 model: Model,
                 iterator: DataIterator,
                 patience: int = 2,
                 val_metric: str = "-loss",
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 cuda_device: int = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 lr_decay: Optional[float] = None,
                 min_lr: Optional[float] = None,
                 no_tqdm: bool = False) -> None:
        """
        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        patience : int, optional (default=2)
            Number of epochs to be patient before early stopping.
        val_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        cuda_device : int, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
            Multi-gpu training is not currently supported, but will be once the
            Pytorch DataParallel API stabilises.
        grad_norm : float, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : PytorchLRScheduler, optional, (default = None)
            A Pytorch learning rate scheduler. The learning rate will be decayed with respect to
            this schedule at the end of each epoch. If you use
            :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`,
            this will use the ``val_metric`` provided to determine if learning has plateaued.
        no_tqdm : ``bool``, optional (default=False)
            We use ``tqdm`` for logging, which will print a nice progress bar that updates in place
            after every batch.  This is nice if you're running training on a local shell, but can
            cause problems with log files from, e.g., a docker image running on kubernetes.  If
            ``no_tqdm`` is ``True``, we will not use tqdm, and instead log batch statistics using
            ``logger.info``, outputting a line at most every 10 seconds.
        """
        self._model = model
        self._iterator = iterator

        self._patience = patience
        self._num_epochs = num_epochs
        self._serialization_dir = serialization_dir
        self._cuda_device = cuda_device
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._lr_decay = lr_decay
        self._min_lr = min_lr

        increase_or_decrease = val_metric[0]
        if increase_or_decrease not in ["+", "-"]:
            raise ConfigurationError("Validation metrics must specify whether they should increase "
                                     "or decrease by pre-pending the metric name with a +/-.")
        self._val_metric = val_metric[1:]
        self._val_metric_decreases = increase_or_decrease == "-"
        self._no_tqdm = no_tqdm

        if self._cuda_device >= 0:
            self._model = self._model.cuda(self._cuda_device)

        self._log_interval = 10  # seconds
        self._summary_interval = 100  # num batches between logging to tensorboard

    def train(self, tasks, task_ordering, validation_interval, max_vals,
              bpp_method, bpp_base,
              optimizer_params, scheduler_params, load_model=1) -> None:
        '''
        Train on tasks.

        Args:
            - tasks (List[Task]):

        Returns: None
        '''

        epoch_counter = 0 # for resuming training
        n_tasks = len(tasks)
        iterator = self._iterator
        patience = self._patience

        # Resume from serialization path if it contains a saved model.
        if self._serialization_dir is not None:
            # Set up tensorboard logging.
            train_logs, val_logs = {}, {}
            for task in tasks:
                train_logs["%s_loss" % task.name] = SummaryWriter(os.path.join(
                    self._serialization_dir, task.name, "train"))
                val_logs["%s_loss" % task.name] = SummaryWriter(os.path.join(
                    self._serialization_dir, task.name, "valid"))
                train_logs[task.val_metric] = SummaryWriter(os.path.join(
                    self._serialization_dir, task.name, "train"))
                val_logs[task.val_metric] = SummaryWriter(os.path.join(
                    self._serialization_dir, task.name, "valid"))
            val_logs["macro_accuracy"] = SummaryWriter(os.path.join(
                self._serialization_dir, "macro_accuracy", "valid"))
            val_logs["micro_accuracy"] = SummaryWriter(os.path.join(
                self._serialization_dir, "micro_accuracy", "valid"))
            if load_model and \
                    any(["model_state_epoch_" in x
                         for x in os.listdir(self._serialization_dir)]):
                logger.info("Loading model from checkpoint.")
                epoch_counter = self._restore_checkpoint()

        if self._grad_clipping is not None:
            # pylint: disable=invalid-unary-operand-type
            clip_function = lambda grad: grad.clamp(
                -self._grad_clipping, self._grad_clipping)
            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

        task_infos = {task.name: {} for task in tasks}
        parameters = [p for p in self._model.parameters() if p.requires_grad]
        if 'proportional' in bpp_method: # for computing n batches per pass
            if 'tr' in bpp_method:
                sizes = [iterator.get_num_batches(task.train_data) for\
                         task in tasks]
                min_size = min(sizes)
                bpps = [size / min_size for size in sizes]
            else:
                sizes = [(idx, iterator.get_num_batches(task.train_data)) for \
                        idx, task in enumerate(tasks)]
                sizes.sort(key=lambda x: x[1])
                bpps = [0] * n_tasks
                for rank, (idx, _) in enumerate(sizes):
                    bpps[idx] = rank + 1
        for task_idx, task in enumerate(tasks):
            task_info = task_infos[task.name]
            n_tr_batches = iterator.get_num_batches(task.train_data)
            n_val_batches = iterator.get_num_batches(task.val_data)
            task_info['n_tr_batches'] = n_tr_batches
            tr_generator = tqdm.tqdm(iterator(task.train_data, num_epochs=None),
                                     disable=self._no_tqdm, total=n_tr_batches)
            task_info['tr_generator'] = tr_generator
            task_info['loss'] = 0.0 # Maybe don't need here
            task_info['total_batches_trained'] = 0
            task_info['n_batches_since_val'] = 0
            task_info['loss'] = 0
            if bpp_method == 'fixed':
                n_batches_per_pass = bpp_base
            elif bpp_method == 'percent_tr':
                n_batches_per_pass = int(math.ceil((1/bpp_base) * n_tr_batches))
            elif bpp_method == 'proportional_rank':
                n_batches_per_pass = int(bpp_base * bpps[task_idx])
            task_info['n_batches_per_pass'] = n_batches_per_pass
            task_info['n_val_batches'] = n_val_batches
            optimizer = Optimizer.from_params(parameters, copy.deepcopy(optimizer_params))
            scheduler = LearningRateScheduler.from_params(optimizer,
                                                          copy.deepcopy(scheduler_params))
            task_info['optimizer'] = optimizer
            task_info['scheduler'] = scheduler
            task_info['last_log'] = time.time()
            logger.info("Task %s: training %d of %d batches per pass", task.name,
                        n_batches_per_pass, n_tr_batches)
            logger.info("\t%d batches, %.3f epochs between validation checks",
                        n_batches_per_pass * validation_interval,
                        n_batches_per_pass * validation_interval / n_tr_batches)
        metric_histories = {task.name: [] for task in tasks}
        metric_histories['all'] = []
        stop_training = {task.name: False for task in tasks}

        # Get ordering on tasks, maybe should vary per pass
        if task_ordering == 'given':
            task_order = range(len(tasks))
        elif task_ordering == 'random':
            task_order = [i for i in range(len(tasks))]
            random.shuffle(task_order)
        elif task_ordering == 'small_to_large':
            task_sizes = [(idx, task_infos[task.name]['n_tr_batches']) \
                          for idx, task in enumerate(tasks)]
            task_sizes.sort(key=lambda x: x[1])
            task_order = [task_idx for task_idx, _ in task_sizes]

        logger.info("Beginning training.")
        should_stop = False
        n_pass = 0
        all_tr_metrics = {}
        while not should_stop:
            self._model.train()
            if task_ordering == 'random':
                random.shuffle(task_order)
            for train_idx, task_idx in enumerate(task_order):
                task = tasks[task_idx]
                if stop_training[task.name]:
                    continue
                task_info = task_infos[task.name]
                tr_generator = task_info['tr_generator']
                optimizer = task_info['optimizer']
                total_batches_trained = task_info['total_batches_trained']
                n_batches_since_val = task_info['n_batches_since_val']
                tr_loss = task_info['loss']
                preds = [0] * 3
                golds = [0] * 3
                n_exs = 0
                for batch in itertools.islice(tr_generator, task_info['n_batches_per_pass']):
                    n_batches_since_val += 1
                    total_batches_trained += 1
                    optimizer.zero_grad()
                    output_dict = self._forward(batch, task=task, for_training=True)
                    assert "loss" in output_dict, "Model must return a dict " \
                                                  "containing a 'loss' key"
                    loss = output_dict["loss"]
                    loss.backward()
                    tr_loss += loss.data.cpu().numpy()
                    preds[0] += torch.sum(torch.eq(output_dict['logits'].max(1)[1], 0.)).cpu().data[0]
                    preds[1] += torch.sum(torch.eq(output_dict['logits'].max(1)[1], 1.)).cpu().data[0]
                    preds[2] += torch.sum(torch.eq(output_dict['logits'].max(1)[1], 2.)).cpu().data[0]
                    golds[0] += torch.sum(torch.eq(batch['label'], 0.)).cpu().data[0]
                    golds[1] += torch.sum(torch.eq(batch['label'], 1.)).cpu().data[0]
                    golds[2] += torch.sum(torch.eq(batch['label'], 2.)).cpu().data[0]
                    n_exs += batch['label'].size()[0]

                    # Gradient regularization and application
                    if self._grad_norm:
                        clip_grad_norm(self._model.parameters(),
                                       self._grad_norm)
                    optimizer.step()

                    # Get metrics for all progress so far, update tqdm
                    task_metrics = task.get_metrics()
                    task_metrics["%s_loss" % task.name] = \
                            float(tr_loss / n_batches_since_val)
                    description = self._description_from_metrics(task_metrics)
                    tr_generator.set_description(description)

                    # Training logging
                    if self._no_tqdm and time.time() - task_info['last_log'] > self._log_interval:
                        logger.info("Task %d/%d: %s, Batch %d (%d): %s", train_idx + 1, n_tasks,
                                    task.name, n_batches_since_val, total_batches_trained,
                                    description)
                        for name, param in self._model.named_parameters():
                            if param.grad is None:
                                continue
                            logger.debug("GRAD MEAN %s: %.7f", name,
                                         param.grad.data.mean())
                            logger.debug("GRAD STD %s: %.7f", name, \
                                         param.grad.data.std())
                        task_info['last_log'] = time.time()

                    # Tensorboard logging
                    if self._serialization_dir and \
                            n_batches_since_val % self._summary_interval == 0:
                        metric = task.val_metric
                        for name, param in self._model.named_parameters():
                            train_logs[metric].add_scalar("PARAMETER_MEAN/" + \
                                    name, param.data.mean(), \
                                    total_batches_trained)
                            train_logs[metric].add_scalar("PARAMETER_STD/" + \
                                    name, param.data.std(), \
                                    total_batches_trained)
                            if param.grad is not None:
                                train_logs[metric].add_scalar("GRAD_MEAN/" + \
                                        name, param.grad.data.mean(), \
                                        total_batches_trained)
                                train_logs[metric].add_scalar("GRAD_STD/" + \
                                        name, param.grad.data.std(), \
                                        total_batches_trained)
                        train_logs[metric].add_scalar("LOSS/loss_train", \
                                task_metrics["%s_loss" % task.name], \
                                n_batches_since_val)

                    # Update training progress on that task
                    task_info['n_batches_since_val'] = n_batches_since_val
                    task_info['total_batches_trained'] = total_batches_trained
                    task_info['loss'] = tr_loss
                logger.info("Predicted %d/%d entailment, %d/%d other, %d/%d not",
                            preds[0], golds[0], preds[1], golds[1], preds[2], golds[2])
                logger.info("Number examples: %d", n_exs)

            # Overall training logging after a pass through data
            for task in tasks:
                if stop_training[task.name]:
                    continue
                task_info = task_infos[task.name]
                task_metrics = task.get_metrics(reset=True)
                for name, value in task_metrics.items():
                    all_tr_metrics["%s_%s" % (task.name, name)] = value
                    all_tr_metrics["%s_loss" % task.name] = \
                            float(task_info['loss'] / task_info['n_batches_since_val'])

            # Validation
            if n_pass % (validation_interval) == 0:
                logger.info("Validating...")
                epoch = int(n_pass / validation_interval)
                val_losses = [0.0] * n_tasks
                n_examples_overall = 0.0
                all_val_metrics = {"macro_accuracy":0.0, "micro_accuracy":0.0}
                self._model.eval()
                for task_idx, task in enumerate(tasks):
                    n_examples = 0.0
                    n_val_batches = task_infos[task.name]['n_val_batches']
                    scheduler = task_infos[task.name]['scheduler']
                    val_generator = iterator(task.val_data, num_epochs=1)
                    val_generator_tqdm = tqdm.tqdm(val_generator,
                                                   disable=self._no_tqdm,
                                                   total=n_val_batches)
                    batch_num = 0
                    for batch in val_generator_tqdm:
                        batch_num += 1
                        val_output_dict = self._forward(batch, task=task,
                                                        for_training=False)
                        loss = val_output_dict["loss"]
                        val_losses[task_idx] += loss.data.cpu().numpy()
                        task_metrics = task.get_metrics()
                        task_metrics["%s_loss" % task.name] = \
                                float(val_losses[task_idx] / batch_num)
                        description = self._description_from_metrics(task_metrics)
                        val_generator_tqdm.set_description(description)
                        if self._no_tqdm and \
                                time.time() - task_info['last_log'] > self._log_interval:
                            logger.info("Batch %d/%d: %s", batch_num, \
                                        n_val_batches, description)
                            task_info['last_log'] = time.time()
                        n_examples += batch['label'].size()[0]

                    # Get task validation metrics and store in all_val_metrics
                    task_metrics = task.get_metrics(reset=True)
                    for name, value in task_metrics.items():
                        all_val_metrics["%s_%s" % (task.name, name)] = value
                    all_val_metrics["%s_loss" % task.name] = \
                            float(val_losses[task_idx] / batch_num)
                    all_val_metrics["micro_accuracy"] += \
                            all_val_metrics["%s_accuracy" % (task.name)] * n_examples
                    all_val_metrics["macro_accuracy"] += \
                            all_val_metrics["%s_accuracy" % (task.name)]
                    n_examples_overall += n_examples

                    # Check if patience ran out and should stop training
                    # No patience check because if we stop training for a single
                    # task, the other tasks will likely cause degredation so we
                    # would want to continue training anyways
                    this_epoch_metric = all_val_metrics[task.val_metric]
                    metric_history = metric_histories[task.name]
                    metric_history.append(this_epoch_metric)

                    # Adjust task-specific learning rate
                    if scheduler is not None:
                        # Grim hack to determine whether the validation metric we
                        # are recording needs to be passed to the scheduler.
                        if isinstance(scheduler,
                                      torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(this_epoch_metric, epoch)
                        else:
                            scheduler.step(epoch)

                # Print all validation metrics
                # Should divide aggregate scores
                all_val_metrics['micro_accuracy'] /= n_examples_overall
                all_val_metrics['macro_accuracy'] /= n_tasks
                logger.info("***** Pass %d / Epoch %d *****", n_pass, epoch)
                for name, value in all_val_metrics.items():
                    logger.info("Statistic: %s", name)
                    if name in all_tr_metrics:
                        logger.info("\ttraining: %3f", all_tr_metrics[name])
                    logger.info("\tvalidation: %3f", value)
                    if self._serialization_dir:
                        if name in all_tr_metrics:
                            train_logs[name].add_scalar(name, all_tr_metrics[name], epoch)
                        val_logs[name].add_scalar(name, value, epoch)

                # Check if should stop based on chosen validation metric
                metric_history = metric_histories['all']
                this_epoch_metric = all_val_metrics[self._val_metric]
                if len(metric_history) > patience:
                    # Is the worst validation performance in past self._patience
                    # epochs is better than current value?
                    if self._val_metric_decreases:
                        should_stop = max(
                            metric_history[-patience:]) <= this_epoch_metric
                    else:
                        should_stop = min(
                            metric_history[-patience:]) >= this_epoch_metric
                    if should_stop:
                        logger.info("Out of patience. Stopping training")
                metric_history.append(this_epoch_metric)

                # Check if should save based on aggregate score
                if self._val_metric_decreases:
                    is_best_so_far = this_epoch_metric == \
                            min(metric_history)
                else:
                    is_best_so_far = this_epoch_metric == \
                            max(metric_history)
                if self._serialization_dir:
                    self._save_checkpoint(epoch, is_best=is_best_so_far)

                # Decay all learning rates based on aggregate score
                # and check if minimum lr has been hit to stop training
                all_stopped = True # check if all tasks are stopped
                for task in tasks:
                    task_info = task_infos[task.name]
                    if not is_best_so_far and self._lr_decay:
                        task_info['optimizer'].param_groups[0]['lr'] *= self._lr_decay
                    if task_info['optimizer'].param_groups[0]['lr'] < self._min_lr:
                        logger.info("Minimum lr hit on %s.", task.name)
                        stop_training[task.name] = True
                    all_stopped = all_stopped and stop_training[task.name]
                if all_stopped:
                    should_stop = True
                    logging.info("All tasks hit minimum lr. Stopping training")

                # Reset training progress after validating
                all_tr_metrics = {}
                for task in tasks:
                    task_infos[task.name]['n_batches_since_val'] = 0
                    task_infos[task.name]['loss'] = 0

                # Check to see if maximum progress hit
                if epoch >= max_vals:
                    logging.info("Maximum number of validations hit. Stopping training")
                    should_stop = True

            n_pass += 1

        # print number of effective epochs trained per task
        logging.info('Stopped training after %d passes, %d validation checks',
                     n_pass, n_pass / validation_interval)
        for task in tasks:
            task_info = task_infos[task.name]
            logging.info('Trained %s for %d batches or %.3f epochs',
                         task.name, task_info['total_batches_trained'],
                         task_info['total_batches_trained'] / task_info['n_tr_batches'])

    def _forward(self, batch: dict, for_training: bool,
                 task=None) -> dict:
        tensor_batch = arrays_to_variables(batch, self._cuda_device, for_training=for_training)
        return self._model.forward(task, **tensor_batch)

    def _description_from_metrics(self, metrics: Dict[str, float]) -> str:
        # pylint: disable=no-self-use
        return ', '.join(["%s: %.4f" % (name, value) for name, value in metrics.items()]) + " ||"

    def _save_checkpoint(self,
                         epoch: int,
                         is_best: Optional[bool] = None) -> None:
        """
        Parameters
        ----------
        epoch : int, required.
            The epoch of training.
        is_best: bool, optional (default = None)
            A flag which causes the model weights at the given epoch to
            be copied to a "best.th" file. The value of this flag should
            be based on some validation metric computed by your model.
        """
        model_path = os.path.join(self._serialization_dir, "model_state_epoch_{}.th".format(epoch))
        model_state = self._model.state_dict()
        torch.save(model_state, model_path)

        training_state = {'epoch': epoch}#, 'optimizer': self._optimizer.state_dict()}
        torch.save(training_state, os.path.join(self._serialization_dir,
                                                "training_state_epoch_{}.th".format(epoch)))
        if is_best:
            logger.info("Best validation performance so far. "
                        "Copying weights to %s/best.th'.", self._serialization_dir)
            shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best.th"))

    def _restore_checkpoint(self) -> int:
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from  model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        Returns
        -------
        epoch: int
            The epoch at which to resume training.
        """
        if not self._serialization_dir:
            raise ConfigurationError("serialization_dir not specified - cannot "
                                     "restore a model without a directory path.")

        serialization_files = os.listdir(self._serialization_dir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        epoch_to_load = max([int(x.split("model_state_epoch_")[-1].strip(".th")) \
                             for x in model_checkpoints])

        model_path = os.path.join(self._serialization_dir,
                                  "model_state_epoch_{}.th".format(epoch_to_load))
        training_state_path = os.path.join(self._serialization_dir,
                                           "training_state_epoch_{}.th".format(epoch_to_load))

        model_state = torch.load(model_path, map_location=device_mapping(self._cuda_device))
        training_state = torch.load(training_state_path)
        self._model.load_state_dict(model_state)
        #self._optimizer.load_state_dict(training_state["optimizer"])
        return training_state["epoch"]

    @classmethod
    def from_params(cls,
                    model: Model,
                    serialization_dir: str,
                    iterator: DataIterator,
                    params: Params) -> 'MultiTaskTrainer':
        '''
        Generator trainer from parameters.
        '''

        patience = params.pop("patience", 2)
        val_metric = params.pop("val_metric", "-loss")
        num_epochs = params.pop("num_epochs", 20)
        cuda_device = params.pop("cuda_device", -1)
        grad_norm = params.pop("grad_norm", None)
        grad_clipping = params.pop("grad_clipping", None)
        lr_decay = params.pop("lr_decay", None)
        min_lr = params.pop("min_lr", None)
        #lr_scheduler_params = params.pop("learning_rate_scheduler", None)

        #if cuda_device >= 0:
        #    model = model.cuda(cuda_device)
        #parameters = [p for p in model.parameters() if p.requires_grad]
        #optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        #if lr_scheduler_params:
        #    scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        #else:
        #    scheduler = None
        no_tqdm = params.pop("no_tqdm", False)

        params.assert_empty(cls.__name__)
        return MultiTaskTrainer(model,
                                #optimizer,
                                iterator,
                                patience=patience,
                                val_metric=val_metric,
                                num_epochs=num_epochs,
                                serialization_dir=serialization_dir,
                                cuda_device=cuda_device,
                                grad_norm=grad_norm,
                                grad_clipping=grad_clipping,
                                lr_decay=lr_decay,
                                min_lr=min_lr,
                                #learning_rate_scheduler=scheduler,
                                no_tqdm=no_tqdm)
