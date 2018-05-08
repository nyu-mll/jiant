"""
A :class:`~allennlp.training.MultiTaskTrainer.MultiTaskTrainer` is responsible
for training a :class:`~allennlp.models.model.Model`.

Typically you might create a configuration file specifying the model and
training parameters and then use :mod:`~allennlp.commands.train`
rather than instantiating a ``MultiTaskTrainer`` yourself.
"""

import os
import ipdb as pdb # pylint: disable=unused-import
import math
import time
import copy
import random
import logging
import itertools

import torch
import torch.optim.lr_scheduler
from torch.nn.utils.clip_grad import clip_grad_norm
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from util import arrays_to_variables, device_mapping
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from tasks import STS14Task, STSBenchmarkTask

def build_trainer(args, trainer_type, model, iterator):
    '''Build a trainer'''
    opt_params = Params({'type': args.optimizer, 'lr': args.lr, 'weight_decay': 1e-5})
    schd_params = Params({'type': 'reduce_on_plateau', 'mode':'max', 'factor': args.lr_decay_factor,
                          'patience': args.task_patience, 'threshold': args.scheduler_threshold,
                          'threshold_mode': 'abs', 'verbose':True})
    train_params = Params({'num_epochs': args.n_epochs, 'cuda_device': args.cuda,
                           'patience': args.patience, 'grad_norm': args.max_grad_norm,
                           'lr_decay': .99, 'min_lr': args.min_lr, 'no_tqdm': args.no_tqdm})
    if trainer_type == 'sampling':
        trainer = SamplingMultiTaskTrainer.from_params(model, args.run_dir, iterator,
                                                       copy.deepcopy(train_params))
    elif trainer_type == 'mtl':
        trainer = MultiTaskTrainer.from_params(model, args.run_dir, iterator,
                                               copy.deepcopy(train_params))
    return trainer, train_params, opt_params, schd_params

def get_task_order(method, tasks, iterator):
    '''Get order to train tasks on'''
    if method == 'given':
        task_order = range(len(tasks))
    elif 'random' in method:
        task_order = [i for i in range(len(tasks))]
        random.shuffle(task_order)
    else:
        task_sizes = [(idx, iterator.get_num_batches(task.train_data)) \
                      for idx, task in enumerate(tasks)]
        task_sizes.sort(key=lambda x: x[1], reverse=bool(method == 'large_to_small'))
        task_order = [task_idx for task_idx, _ in task_sizes]
    return task_order

class MultiTaskTrainer:
    def __init__(self, model, iterator, patience=2, num_epochs=20, serialization_dir=None, cuda_device=-1,
                 grad_norm=None, grad_clipping=None, lr_decay=None, min_lr=None, no_tqdm=False):
        """ Parameters
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
        patience , optional (default=2)
            Number of epochs to be patient before early stopping.
        val_metric , optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        num_epochs , optional (default = 20)
            Number of training epochs.
        serialization_dir , optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        cuda_device , optional (default = -1)
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

        self._task_infos = None
        self._metric_infos = None

        self._no_tqdm = no_tqdm
        self._log_interval = 10  # seconds
        self._summary_interval = 100  # num batches between logging to tensorboard
        if self._cuda_device >= 0:
            self._model = self._model.cuda(self._cuda_device)

    def _check_history(self, metric_history, cur_score, should_decrease=False):
        '''
        Given a task, the history of the performance on that task,
        and the current score, check if current score is
        best so far and if out of patience.
        '''
        patience = self._patience + 1
        best_fn = min if should_decrease else max
        best_score = best_fn(metric_history)
        if best_score == cur_score:
            best_so_far = metric_history.index(best_score) == len(metric_history) - 1
        else:
            best_so_far = False

        out_of_patience = False
        if len(metric_history) > patience:
            if should_decrease:
                out_of_patience = max(metric_history[-patience:]) <= cur_score
            else:
                out_of_patience = min(metric_history[-patience:]) >= cur_score

        if best_so_far and out_of_patience: # then something is up
            pdb.set_trace()

        return best_so_far, out_of_patience


    def train(self, tasks, task_ordering, validation_interval, max_vals,
              bpp_method, bpp_base,
              train_params, optimizer_params, scheduler_params, load_model=1):
        '''
        Train on tasks.

        Args:
            - tasks (List[Task]):

        Returns: None
        '''

        n_pass, should_stop = 0, False
        n_tasks = len(tasks)
        iterator = self._iterator
        parameters = train_params
        task_order = get_task_order(task_ordering, tasks, iterator)

        if 'proportional' in bpp_method: # for computing n batches per pass
            if 'tr' in bpp_method:
                sizes = [iterator.get_num_batches(task.train_data) for task in tasks]
                min_size = min(sizes)
                bpps = [size / min_size for size in sizes]
            else:
                sizes = [(idx, iterator.get_num_batches(task.train_data)) for \
                        idx, task in enumerate(tasks)]
                sizes.sort(key=lambda x: x[1])
                bpps = [0] * n_tasks
                for rank, (idx, _) in enumerate(sizes):
                    bpps[idx] = rank + 1

        # Task bookkeeping
        task_infos = {task.name: {} for task in tasks}
        for task_idx, task in enumerate(tasks):
            task_info = task_infos[task.name]
            n_tr_batches = iterator.get_num_batches(task.train_data)
            n_val_batches = iterator.get_num_batches(task.val_data)
            task_info['n_tr_batches'] = n_tr_batches
            tr_generator = tqdm.tqdm(iterator(task.train_data, num_epochs=None,
                                              cuda_device=self._cuda_device),
                                     disable=self._no_tqdm, total=n_tr_batches)
            task_info['tr_generator'] = tr_generator
            task_info['loss'] = 0.0 # Maybe don't need here
            task_info['total_batches_trained'] = 0
            task_info['n_batches_since_val'] = 0
            if bpp_method == 'fixed':
                n_batches_per_pass = bpp_base
            elif bpp_method == 'percent_tr':
                n_batches_per_pass = int(math.ceil((1/bpp_base) * n_tr_batches))
            elif bpp_method == 'proportional_rank':
                n_batches_per_pass = int(bpp_base * bpps[task_idx])
            task_info['n_batches_per_pass'] = n_batches_per_pass
            task_info['n_val_batches'] = n_val_batches
            task_info['optimizer'] = Optimizer.from_params(parameters,
                                                           copy.deepcopy(optimizer_params))
            task_info['scheduler'] = LearningRateScheduler.from_params(task_info['optimizer'],
                                                                       copy.deepcopy(scheduler_params))
            task_info['stopped'] = False
            task_info['last_log'] = time.time()
            logger.info("Task %s: training %d of %d batches per pass", task.name,
                        n_batches_per_pass, n_tr_batches)
            logger.info("\t%d batches, %.3f epochs between validation checks",
                        n_batches_per_pass * validation_interval,
                        n_batches_per_pass * validation_interval / n_tr_batches)

        # Metric bookkeeping
        all_metrics = [task.val_metric for task in tasks] + ['micro_accuracy', 'macro_accuracy']
        metric_infos = {metric: {'hist': [], 'stopped': False, 'best': (-1, {})} for \
                        metric in all_metrics}

        self._task_infos = task_infos
        self._metric_infos = metric_infos

        # Resume from serialization path if it contains a saved model.
        if self._serialization_dir is not None:
            # Set up tensorboard logging.
            if load_model and any(["model_state_epoch_" in x
                                   for x in os.listdir(self._serialization_dir)]):
                n_pass, should_stop = self._restore_checkpoint()
                logger.info("Loaded model from checkpoint. Starting at pass %d", n_pass)

        if self._grad_clipping is not None:
            # pylint: disable=invalid-unary-operand-type
            clip_function = lambda grad: grad.clamp(
                -self._grad_clipping, self._grad_clipping)
            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

        logger.info("Beginning training.")
        logger.info("\tTask order: %s", ", ".join([tasks[idx].name for idx in task_order]))
        all_tr_metrics = {}
        while not should_stop:
            self._model.train()
            if task_ordering == 'random_per_pass':
                random.shuffle(task_order)
            for train_idx, task_idx in enumerate(task_order):
                task = tasks[task_idx]
                task_info = task_infos[task.name]
                if task_info['stopped']:
                    continue
                tr_generator = task_info['tr_generator']
                optimizer = task_info['optimizer']
                total_batches_trained = task_info['total_batches_trained']
                n_batches_since_val = task_info['n_batches_since_val']
                tr_loss = task_info['loss']
                for batch in itertools.islice(tr_generator, task_info['n_batches_per_pass']):
                    n_batches_since_val += 1
                    total_batches_trained += 1
                    optimizer.zero_grad()
                    output_dict = self._forward(batch, task=task, for_training=True)
                    assert "loss" in output_dict, "Model must return a dict containing a 'loss' key"
                    loss = output_dict["loss"]
                    loss.backward()
                    tr_loss += loss.data.cpu().numpy()

                    # Gradient regularization and application
                    if self._grad_norm:
                        clip_grad_norm(self._model.parameters(), self._grad_norm)
                    optimizer.step()

                    # Get metrics for all progress so far, update tqdm
                    task_metrics = task.get_metrics()
                    task_metrics["%s_loss" % task.name] = float(tr_loss / n_batches_since_val)
                    description = self._description_from_metrics(task_metrics)
                    tr_generator.set_description(description)

                    # Training logging
                    if self._no_tqdm and time.time() - task_info['last_log'] > self._log_interval:
                        logger.info("Task %d/%d: %s, Batch %d (%d): %s", train_idx + 1, n_tasks,
                                    task.name, n_batches_since_val, total_batches_trained, description)
                        for name, param in self._model.named_parameters():
                            logger.debug("PARAM MEAN %s: %.7f", name, param.data.mean())
                            logger.debug("PARAM STD %s: %.7f", name, param.data.std())
                            if param.grad is None:
                                continue
                            logger.debug("GRAD MEAN %s: %.7f", name, param.grad.data.mean())
                            logger.debug("GRAD STD %s: %.7f", name, param.grad.data.std())
                        task_info['last_log'] = time.time()

                    # Update training progress on that task
                    task_info['n_batches_since_val'] = n_batches_since_val
                    task_info['total_batches_trained'] = total_batches_trained
                    task_info['loss'] = tr_loss

            # Overall training logging after a pass through data
            for task in tasks:
                task_info = task_infos[task.name]
                if task_info['stopped']:
                    continue
                task_metrics = task.get_metrics(reset=True)
                for name, value in task_metrics.items():
                    all_tr_metrics["%s_%s" % (task.name, name)] = value
                all_tr_metrics["%s_loss" % task.name] = \
                        float(task_info['loss'] / task_info['n_batches_since_val'])
            n_pass += 1

            # Validation
            if n_pass % (validation_interval) == 0:
                logger.info("Validating...")
                epoch = int(n_pass / validation_interval)
                val_losses = [0.0] * n_tasks
                n_examples_overall = 0.0
                should_save = False
                all_val_metrics = {"macro_accuracy":0.0, "micro_accuracy":0.0}
                self._model.eval()
                for task_idx, task in enumerate(tasks):
                    n_examples = 0.0
                    n_val_batches = task_infos[task.name]['n_val_batches']
                    scheduler = task_infos[task.name]['scheduler']
                    val_generator = iterator(task.val_data, num_epochs=1, cuda_device=self._cuda_device)
                    val_generator_tqdm = tqdm.tqdm(val_generator, disable=self._no_tqdm,
                                                   total=n_val_batches)
                    batch_num = 0
                    for batch in val_generator_tqdm:
                        batch_num += 1
                        val_output_dict = self._forward(batch, task=task, for_training=False)
                        loss = val_output_dict["loss"]
                        val_losses[task_idx] += loss.data.cpu().numpy()
                        task_metrics = task.get_metrics()
                        task_metrics["%s_loss" % task.name] = \
                                float(val_losses[task_idx] / batch_num)
                        description = self._description_from_metrics(task_metrics)
                        val_generator_tqdm.set_description(description)
                        if self._no_tqdm and \
                                time.time() - task_info['last_log'] > self._log_interval:
                            logger.info("Batch %d/%d: %s", batch_num, n_val_batches, description)
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

                    # If not out of patience,
                    # check if should save based on aggregate score
                    # do a patience check, adjust learning rate
                    if not metric_infos[task.val_metric]['stopped']:
                        this_epoch_metric = all_val_metrics[task.val_metric]
                        metric_history = metric_infos[task.val_metric]['hist']
                        metric_history.append(this_epoch_metric)
                        is_best_so_far, out_of_patience = \
                                self._check_history(metric_history, this_epoch_metric,
                                                    task.val_metric_decreases)
                        if is_best_so_far:
                            logger.info("Best model found for %s.", task.name)
                            metric_infos[task.val_metric]['best'] = (epoch, all_val_metrics)
                            should_save = True
                        if out_of_patience:
                            metric_infos[task.val_metric]['stopped'] = True
                            logger.info("Out of patience. Stopped tracking %s", task.name)
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(this_epoch_metric, epoch)
                    else:
                        scheduler.step(epoch)

                # Print all validation metrics
                all_val_metrics['micro_accuracy'] /= n_examples_overall
                all_val_metrics['macro_accuracy'] /= n_tasks
                logger.info("***** Pass %d / Epoch %d *****", n_pass, epoch)
                for name, value in all_val_metrics.items():
                    logger.info("Statistic: %s", name)
                    if name in all_tr_metrics:
                        logger.info("\ttraining: %3f", all_tr_metrics[name])
                    logger.info("\tvalidation: %3f", value)

                # Track macro and micro
                for task in ['micro', 'macro']:
                    metric = "%s_accuracy" % task # not really accuracy
                    if metric_infos[metric]['stopped']:
                        continue
                    this_epoch_metric = all_val_metrics[metric]
                    metric_history = metric_infos[metric]['hist']
                    metric_history.append(this_epoch_metric)
                    is_best_so_far, out_of_patience = \
                            self._check_history(metric_history, this_epoch_metric)
                    if is_best_so_far:
                        logger.info("Best model found for %s.", task)
                        metric_infos[metric]['best'] = (epoch, all_val_metrics)
                        should_save = True
                    if out_of_patience:
                        metric_infos[metric]['stopped'] = True
                        logger.info("Out of patience. Stopped tracking %s", task)

                # Reset training progress after validating
                stop_tr = True
                stop_val = True
                for task in tasks:
                    task_info = task_infos[task.name]
                    if task_info['optimizer'].param_groups[0]['lr'] < self._min_lr:
                        logger.info("Minimum lr hit on %s.", task.name)
                        task_info['stopped'] = True
                    stop_tr = stop_tr and task_info['stopped']
                    stop_val = stop_val and metric_infos[task.val_metric]['stopped']
                    task_info['n_batches_since_val'] = 0
                    task_info['loss'] = 0
                stop_val = stop_val and metric_infos['micro_accuracy']['stopped'] and \
                            metric_infos['macro_accuracy']['stopped']
                all_tr_metrics = {}

                # Check to see if should stop
                if stop_tr:
                    should_stop = True
                    logging.info("All tasks hit minimum lr. Stopping training.")
                if stop_val:
                    should_stop = True
                    logging.info("All metrics ran out of patience. Stopping training.")
                if epoch >= max_vals:
                    logging.info("Maximum number of validations hit. Stopping training.")
                    should_stop = True
                self._metric_infos = metric_infos
                self._task_infos = task_infos
                if should_save:
                    self._save_checkpoint({"pass": n_pass, "should_stop": should_stop})


        # print number of effective epochs trained per task
        logging.info('Stopped training after %d passes, %d validation checks',
                     n_pass, n_pass / validation_interval)
        return_dict = {}
        for task in tasks:
            task_info = task_infos[task.name]
            logging.info('Trained %s for %d batches or %.3f epochs',
                         task.name, task_info['total_batches_trained'],
                         task_info['total_batches_trained'] / task_info['n_tr_batches'])
            return_dict[task.name] = metric_infos[task.val_metric]['best'][0] * validation_interval
        return_dict['micro'] = metric_infos['micro_accuracy']['best'][0] * validation_interval
        return_dict['macro'] = metric_infos['macro_accuracy']['best'][0] * validation_interval
        logging.info('***** VALIDATION RESULTS *****')
        for metric in metric_infos.keys():
            best_epoch, epoch_metrics = metric_infos[metric]['best']
            all_metrics_str = ', '.join(['%s: %.5f' % (metric, score) for \
                                         metric, score in epoch_metrics.items()])
            logging.info('%s, %d, %s', metric, best_epoch, all_metrics_str)
        return return_dict

    def _forward(self, batch, for_training, task=None):
        # tensor_batch = arrays_to_variables(batch, self._cuda_device, for_training=for_training)
        tensor_batch = batch
        return self._model.forward(task, **tensor_batch)

    def _description_from_metrics(self, metrics):
        # pylint: disable=no-self-use
        return ', '.join(["%s: %.4f" % (name, value) for name, value in metrics.items()]) + " ||"

    def _save_checkpoint(self, training_state):
        """
        Parameters
        ----------
        epoch , required.
            The epoch of training.
        is_best, optional (default = None)
            A flag which causes the model weights at the given epoch to
            be copied to a "best.th" file. The value of this flag should
            be based on some validation metric computed by your model.
        """
        epoch = training_state["pass"]
        #model_path = os.path.join(self._serialization_dir, "{}_best.th".format(task))
        model_path = os.path.join(self._serialization_dir, "model_state_epoch_{}.th".format(epoch))
        model_state = self._model.state_dict()
        torch.save(model_state, model_path)

        torch.save(training_state, os.path.join(self._serialization_dir,
                                                "training_state_epoch_{}.th".format(epoch)))

        task_states = {}
        for task_name, task_info in self._task_infos.items():
            task_states[task_name] = {}
            task_states[task_name]['optimizer'] = task_info['optimizer'].state_dict()
            task_states[task_name]['scheduler'] = task_info['scheduler']
            sched = task_info['scheduler']
            sched_params = {'best': sched.best, 'num_bad_epochs': sched.num_bad_epochs,
                            'cooldown_counter': sched.cooldown_counter}
            task_states[task_name]['scheduler'] = sched_params
            task_states[task_name]['total_batches_trained'] = task_info['total_batches_trained']
            task_states[task_name]['stopped'] = task_info['stopped']
        torch.save(task_states, os.path.join(self._serialization_dir,
                                             "task_state_epoch_{}.th".format(epoch)))

        metric_states = {}
        for metric_name, metric_info in self._metric_infos.items():
            metric_states[metric_name] = {}
            metric_states[metric_name]['hist'] = metric_info['hist']
            metric_states[metric_name]['stopped'] = metric_info['stopped']
            metric_states[metric_name]['best'] = metric_info['best']
        torch.save(metric_states, os.path.join(self._serialization_dir,
                                             "metric_state_epoch_{}.th".format(epoch)))
        logging.info("Saved files to %s", self._serialization_dir)

    def _restore_checkpoint(self):
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from  model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        Returns
        -------
        epoch
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
        task_state_path = os.path.join(self._serialization_dir,
                                       "task_state_epoch_{}.th".format(epoch_to_load))
        metric_state_path = os.path.join(self._serialization_dir,
                                         "metric_state_epoch_{}.th".format(epoch_to_load))

        model_state = torch.load(model_path, map_location=device_mapping(self._cuda_device))
        training_state = torch.load(training_state_path)
        task_states = torch.load(task_state_path)
        metric_states = torch.load(metric_state_path)

        self._model.load_state_dict(model_state)
        for task_name, task_state in task_states.items():
            self._task_infos[task_name]['optimizer'].load_state_dict(task_state['optimizer'])
            for param, val in task_state['scheduler'].items():
                setattr(self._task_infos[task_name]['scheduler'], param, val)
            self._task_infos[task_name]['total_batches_trained'] = task_state['total_batches_trained']
            self._task_infos[task_name]['stopped'] = task_state['stopped']
        for metric_name, metric_state in metric_states.items():
            self._metric_infos[metric_name]['hist'] = metric_state['hist']
            self._metric_infos[metric_name]['stopped'] = metric_state['stopped']
            self._metric_infos[metric_name]['best'] = metric_state['best']
        return training_state["pass"], training_state["should_stop"]

    @classmethod
    def from_params(cls, model, serialization_dir, iterator, params):
        ''' Generator trainer from parameters.  '''

        patience = params.pop("patience", 2)
        num_epochs = params.pop("num_epochs", 20)
        cuda_device = params.pop("cuda_device", -1)
        grad_norm = params.pop("grad_norm", None)
        grad_clipping = params.pop("grad_clipping", None)
        lr_decay = params.pop("lr_decay", None)
        min_lr = params.pop("min_lr", None)
        no_tqdm = params.pop("no_tqdm", False)

        params.assert_empty(cls.__name__)
        return MultiTaskTrainer(model,
                                iterator,
                                patience=patience,
                                num_epochs=num_epochs,
                                serialization_dir=serialization_dir,
                                cuda_device=cuda_device,
                                grad_norm=grad_norm,
                                grad_clipping=grad_clipping,
                                lr_decay=lr_decay,
                                min_lr=min_lr,
                                no_tqdm=no_tqdm)

class SamplingMultiTaskTrainer:
    def __init__(self, model, iterator, patience=2, num_epochs=20, max_vals=50,
                 serialization_dir=None, cuda_device=-1,
                 grad_norm=None, grad_clipping=None, lr_decay=None, min_lr=None, no_tqdm=False):
        """ Parameters
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
        patience , optional (default=2)
            Number of epochs to be patient before early stopping.
        val_metric , optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        num_epochs , optional (default = 20)
            Number of training epochs.
        serialization_dir , optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        cuda_device , optional (default = -1)
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
        self._max_vals = max_vals
        self._serialization_dir = serialization_dir
        self._cuda_device = cuda_device
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._lr_decay = lr_decay
        self._min_lr = min_lr

        self._task_infos = None
        self._metric_infos = None

        self._no_tqdm = no_tqdm
        self._log_interval = 10  # seconds
        self._summary_interval = 100  # num batches between logging to tensorboard
        if self._cuda_device >= 0:
            self._model = self._model.cuda(self._cuda_device)

    def _check_history(self, metric_history, cur_score, should_decrease=False):
        '''
        Given a task, the history of the performance on that task,
        and the current score, check if current score is
        best so far and if out of patience.
        '''
        patience = self._patience + 1
        best_fn = min if should_decrease else max
        best_score = best_fn(metric_history)
        if best_score == cur_score:
            best_so_far = metric_history.index(best_score) == len(metric_history) - 1
        else:
            best_so_far = False

        out_of_patience = False
        if len(metric_history) > patience:
            if should_decrease:
                out_of_patience = max(metric_history[-patience:]) <= cur_score
            else:
                out_of_patience = min(metric_history[-patience:]) >= cur_score

        if best_so_far and out_of_patience:
            pdb.set_trace()

        return best_so_far, out_of_patience


    def _setup_training(self, tasks, train_params, optimizer_params, scheduler_params, iterator):
        # Task bookkeeping
        task_infos = {task.name: {} for task in tasks}
        for task in tasks:
            task_info = task_infos[task.name]
            tr_generator = iterator(task.train_data, num_epochs=None, cuda_device=self._cuda_device)
            task_info['n_tr_batches'] = iterator.get_num_batches(task.train_data)
            task_info['tr_generator'] = tr_generator
            task_info['loss'] = 0.0
            task_info['total_batches_trained'] = 0
            task_info['n_batches_since_val'] = 0
            task_info['optimizer'] = Optimizer.from_params(train_params,
                                                           copy.deepcopy(optimizer_params))
            task_info['scheduler'] = LearningRateScheduler.from_params(task_info['optimizer'],
                                                                       copy.deepcopy(scheduler_params))
            task_info['stopped'] = False
            task_info['last_log'] = time.time()
        # Metric bookkeeping
        all_metrics = [task.val_metric for task in tasks] + ['micro_accuracy', 'macro_accuracy']
        metric_infos = {metric: {'hist': [], 'stopped': False, 'best': (-1, {})} for \
                        metric in all_metrics}
        self._task_infos = task_infos
        self._metric_infos = metric_infos
        return task_infos, metric_infos


    def train(self, tasks, validation_interval, n_batches_per_pass,
              weighting_method, scaling_method,
              train_params, optimizer_params, scheduler_params,
              shared_optimizer=0, load_model=1):

        iterator = self._iterator
        task_infos, metric_infos = self._setup_training(tasks, train_params, optimizer_params,
                                                        scheduler_params, iterator)
        if shared_optimizer:
            g_optimizer = Optimizer.from_params(train_params, copy.deepcopy(optimizer_params))
            g_scheduler = LearningRateScheduler.from_params(g_optimizer, copy.deepcopy(scheduler_params))
        else:
            g_optimizer, g_scheduler = None, None
        self._g_optimizer = g_optimizer
        self._g_scheduler = g_scheduler

        n_pass, should_stop = 0, False # define these here b/c they might get overridden on load
        if self._serialization_dir is not None: # Resume from serialization path
            if load_model \
                    and any(["model_state_epoch_" in x for x in os.listdir(self._serialization_dir)]):
                n_pass, should_stop = self._restore_checkpoint()
                logger.info("Loaded model from checkpoint. Starting at pass %d", n_pass)

        if self._grad_clipping is not None: # pylint: disable=invalid-unary-operand-type
            clip_function = lambda grad: grad.clamp(-self._grad_clipping, self._grad_clipping)
            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

        if weighting_method == 'uniform':
            sample_weights = [1] * len(tasks)
        elif weighting_method == 'proportional':
            sample_weights = [task_infos[task.name]['n_tr_batches'] for task in tasks]
            max_weight = max(sample_weights)
            min_weight = min(sample_weights)
        samples = random.choices(tasks, weights=sample_weights, k=validation_interval)

        logger.info("Beginning training.")
        all_tr_metrics = {}
        while not should_stop:
            self._model.train()

            # randomly select a task
            #task = random.choice(tasks)
            task =  samples[n_pass % (validation_interval)]
            task_info = task_infos[task.name]
            if task_info['stopped']:
                continue
            tr_generator = task_info['tr_generator']
            optimizer = g_optimizer if shared_optimizer else task_info['optimizer']
            total_batches_trained = task_info['total_batches_trained']
            n_batches_since_val = task_info['n_batches_since_val']
            tr_loss = task_info['loss']
            for batch in itertools.islice(tr_generator, n_batches_per_pass):
                n_batches_since_val += 1
                total_batches_trained += 1
                optimizer.zero_grad()
                output_dict = self._forward(batch, task=task, for_training=True)
                assert "loss" in output_dict, "Model must return a dict containing a 'loss' key"
                loss = output_dict["loss"] # optionally scale loss
                if scaling_method == 'unit' and weighting_method == 'proportional':
                    loss /= task_info['n_tr_batches']
                elif scaling_method == 'max' and weighting_method == 'proportional':
                    loss *= (max_weight / task_info['n_tr_batches'])
                elif scaling_method == 'min' and weighting_method == 'proportional':
                    loss *= (min_weight / task_info['n_tr_batches'])
                loss.backward()
                tr_loss += loss.data.cpu().numpy()

                # Gradient regularization and application
                if self._grad_norm:
                    clip_grad_norm(self._model.parameters(), self._grad_norm)
                optimizer.step()

                n_pass += 1 # update per batch

            # Update training progress on that task
            task_info['n_batches_since_val'] = n_batches_since_val
            task_info['total_batches_trained'] = total_batches_trained
            task_info['loss'] = tr_loss

            # Intermediate logging
            if time.time() - task_info['last_log'] > self._log_interval:
                task_metrics = task.get_metrics()
                task_metrics["%s_loss" % task.name] = tr_loss / n_batches_since_val
                description = self._description_from_metrics(task_metrics)
                logger.info("Update %d: task %s, batch %d (%d): %s", n_pass,
                            task.name, n_batches_since_val, total_batches_trained, description)
                task_info['last_log'] = time.time()

            # Validation
            if n_pass % (validation_interval) == 0:
                epoch = int(n_pass / validation_interval)
                logger.info("***** Pass %d / Epoch %d *****", n_pass, epoch)
                # Get metrics for all training progress so far
                for task in tasks:
                    task_info = task_infos[task.name]
                    n_batches_since_val = task_info['n_batches_since_val']
                    if n_batches_since_val > 0:
                        task_metrics = task.get_metrics(reset=True)
                        for name, value in task_metrics.items():
                            all_tr_metrics["%s_%s" % (task.name, name)] = value
                        all_tr_metrics["%s_loss" % task.name] = \
                                float(task_info['loss'] / n_batches_since_val)
                    else:
                        all_tr_metrics["%s_loss" % task.name] = 0.0
                    logger.info("%s: trained on %d batches, %.3f epochs", task.name,
                                n_batches_since_val, n_batches_since_val / task_info['n_tr_batches'])

                # Validate
                logger.info("Validating...")
                all_val_metrics, should_save, task_infos, metric_infos = \
                        self._validate(epoch, tasks, task_infos, metric_infos, iterator, g_scheduler)

                # Check stopping conditions
                should_stop, task_infos, metric_infos = \
                        self._check_stop(epoch, tasks, task_infos, metric_infos, g_optimizer)

                # Log results
                for name, value in all_val_metrics.items():
                    logger.info("Statistic: %s", name)
                    if name in all_tr_metrics:
                        logger.info("\ttraining: %3f", all_tr_metrics[name])
                    logger.info("\tvalidation: %3f", value)

                self._metric_infos = metric_infos
                self._task_infos = task_infos
                all_tr_metrics = {}
                samples = random.choices(tasks, weights=sample_weights, k=validation_interval)

                if should_save:
                    self._save_checkpoint({"epoch": epoch, "should_stop": should_stop})

        logging.info('Stopped training after %d validation checks', n_pass / validation_interval)
        return self._aggregate_results(tasks, task_infos, metric_infos)#, validation_interval)

    def _aggregate_results(self, tasks, task_infos, metric_infos):
        ''' Ad hoc helper function to print results after finishing training '''
        results = {}
        for task in tasks:
            task_info = task_infos[task.name]
            logging.info('Trained %s for %d batches or %.3f epochs',
                         task.name, task_info['total_batches_trained'],
                         task_info['total_batches_trained'] / task_info['n_tr_batches'])
            results[task.name] = metric_infos[task.val_metric]['best'][0]# * validation_interval
        results['micro'] = metric_infos['micro_accuracy']['best'][0]# * validation_interval
        results['macro'] = metric_infos['macro_accuracy']['best'][0]# * validation_interval
        logging.info('***** VALIDATION RESULTS *****')
        for metric in metric_infos.keys():
            best_epoch, epoch_metrics = metric_infos[metric]['best']
            all_metrics_str = ', '.join(['%s: %.5f' % (metric, score) for \
                                         metric, score in epoch_metrics.items()])
            logging.info('%s, %d, %s', metric, best_epoch, all_metrics_str)
        return results

    def _validate(self, epoch, tasks, task_infos, metric_infos, iterator, g_scheduler):
        ''' Validate on all tasks and return the results and whether to save this epoch or not '''
        self._model.eval()
        all_val_metrics = {("%s_loss" % task.name): 0.0 for task in tasks}
        all_val_metrics["macro_accuracy"] = 0.0
        all_val_metrics["micro_accuracy"] = 0.0
        n_examples_overall = 0.0

        # Get validation numbers for each task
        for task in tasks:
            n_examples = 0.0
            task_info = task_infos[task.name]
            val_generator = iterator(task.val_data, num_epochs=1, cuda_device=self._cuda_device)
            n_val_batches = iterator.get_num_batches(task.val_data)
            all_val_metrics["%s_loss" % task.name] = 0.0
            batch_num = 0
            for batch in val_generator:
                batch_num += 1
                val_output_dict = self._forward(batch, task=task, for_training=False)
                loss = val_output_dict["loss"]
                all_val_metrics["%s_loss" % task.name] += loss.data.cpu().numpy()

                # Logging
                if time.time() - task_info['last_log'] > self._log_interval:
                    task_metrics = task.get_metrics()
                    task_metrics["%s_loss" % task.name] = all_val_metrics["%s_loss" % task.name] / batch_num
                    description = self._description_from_metrics(task_metrics)
                    logger.info("Batch %d/%d: %s", batch_num, n_val_batches, description)
                    task_info['last_log'] = time.time()
                n_examples += batch['label'].size()[0]
            assert batch_num == n_val_batches

            # Get task validation metrics and store in all_val_metrics
            task_metrics = task.get_metrics(reset=True)
            for name, value in task_metrics.items():
                all_val_metrics["%s_%s" % (task.name, name)] = value
            all_val_metrics["%s_loss" % task.name] /= n_val_batches
            all_val_metrics["micro_accuracy"] += \
                    all_val_metrics["%s_accuracy" % (task.name)] * n_examples
            all_val_metrics["macro_accuracy"] += \
                    all_val_metrics["%s_accuracy" % (task.name)]
            n_examples_overall += n_examples

            # Reset training progress
            task_info['n_batches_since_val'] = 0
            task_info['loss'] = 0

        all_val_metrics['micro_accuracy'] /= n_examples_overall
        all_val_metrics['macro_accuracy'] /= len(tasks)

        # Track per task patience
        should_save = False # whether to save this epoch or not
        for task in tasks + ['micro', 'macro']:
            if task in ['micro', 'macro']:
                metric = "%s_accuracy" % task # not really accuracy
            else:
                metric = task.val_metric
                task = task.name
            if metric_infos[metric]['stopped']:
                continue
            this_epoch_metric = all_val_metrics[metric]
            metric_history = metric_infos[metric]['hist']
            metric_history.append(this_epoch_metric)
            is_best_so_far, out_of_patience = \
                    self._check_history(metric_history, this_epoch_metric)
            if is_best_so_far:
                logger.info("Best model found for %s.", task)
                metric_infos[metric]['best'] = (epoch, all_val_metrics)
                should_save = True
            if out_of_patience:
                metric_infos[metric]['stopped'] = True
                logger.info("Out of patience. Stopped tracking %s", task)

            if hasattr(task, 'name') and g_scheduler is None: # might be "is not None"?
                scheduler = task_infos[task.name]['scheduler']
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(this_epoch_metric, epoch)
                else:
                    scheduler.step(epoch)
            elif g_scheduler is not None and task == 'macro':
                scheduler = g_scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(this_epoch_metric, epoch)
                else:
                    scheduler.step(epoch)

        return all_val_metrics, should_save, task_infos, metric_infos

    def _check_stop(self, epoch, tasks, task_infos, metric_infos, g_optimizer):
        ''' Check to see if should stop '''
        if g_optimizer is None:
            stop_tr = True
            for task in tasks:
                task_info = task_infos[task.name]
                if task_info['optimizer'].param_groups[0]['lr'] < self._min_lr:
                    logger.info("Minimum lr hit on %s.", task.name)
                    task_info['stopped'] = True
                stop_tr = stop_tr and task_info['stopped']
                #stop_val = stop_val and metric_infos[task.val_metric]['stopped']
        else:
            if g_optimizer.param_groups[0]['lr'] < self._min_lr:
                logger.info("Minimum lr hit.")
                stop_tr = True
            else:
                stop_tr = False

        #stop_val = stop_val and metric_infos['micro_accuracy']['stopped'] and \
        #            metric_infos['macro_accuracy']['stopped']
        stop_val = metric_infos['macro_accuracy']['stopped']

        should_stop = False
        if stop_tr:
            should_stop = True
            logging.info("All tasks hit minimum lr. Stopping training.")
        if stop_val:
            should_stop = True
            logging.info("All metrics ran out of patience. Stopping training.")
        if epoch >= self._max_vals:
            logging.info("Maximum number of validations hit. Stopping training.")
            should_stop = True

        return should_stop, task_infos, metric_infos

    def _forward(self, batch, for_training, task=None):
        # tensor_batch = arrays_to_variables(batch, self._cuda_device, for_training=for_training)
        tensor_batch = batch
        return self._model.forward(task, **tensor_batch)

    def _description_from_metrics(self, metrics):
        # pylint: disable=no-self-use
        return ', '.join(["%s: %.4f" % (name, value) for name, value in metrics.items()]) + " ||"

    def _save_checkpoint(self, training_state):
        """
        Parameters
        ----------
        epoch , required.
            The epoch of training.
        is_best, optional (default = None)
            A flag which causes the model weights at the given epoch to
            be copied to a "best.th" file. The value of this flag should
            be based on some validation metric computed by your model.
        """
        epoch = training_state["epoch"]
        model_path = os.path.join(self._serialization_dir, "model_state_epoch_{}.th".format(epoch))
        model_state = self._model.state_dict()
        torch.save(model_state, model_path)

        torch.save(training_state, os.path.join(self._serialization_dir,
                                                "training_state_epoch_{}.th".format(epoch)))

        task_states = {}
        for task_name, task_info in self._task_infos.items():
            task_states[task_name] = {}
            task_states[task_name]['total_batches_trained'] = task_info['total_batches_trained']
            task_states[task_name]['stopped'] = task_info['stopped']
            task_states[task_name]['optimizer'] = task_info['optimizer'].state_dict()
            sched = task_info['scheduler']
            sched_params = {'best': sched.best, 'num_bad_epochs': sched.num_bad_epochs,
                            'cooldown_counter': sched.cooldown_counter}
            task_states[task_name]['scheduler'] = sched_params
        task_states['global'] = {}
        task_states['global']['optimizer'] = self._g_optimizer.state_dict() if \
                self._g_optimizer is not None else None
        if self._g_scheduler is not None:
            sched = self._g_scheduler
            sched_params = {'best': sched.best, 'num_bad_epochs': sched.num_bad_epochs,
                        'cooldown_counter': sched.cooldown_counter}
            task_states['global']['scheduler'] = sched_params
        else:
            task_states['global']['scheduler'] = None
        torch.save(task_states, os.path.join(self._serialization_dir,
                                             "task_state_epoch_{}.th".format(epoch)))

        metric_states = {}
        for metric_name, metric_info in self._metric_infos.items():
            metric_states[metric_name] = {}
            metric_states[metric_name]['hist'] = metric_info['hist']
            metric_states[metric_name]['stopped'] = metric_info['stopped']
            metric_states[metric_name]['best'] = metric_info['best']
        torch.save(metric_states, os.path.join(self._serialization_dir,
                                             "metric_state_epoch_{}.th".format(epoch)))
        logging.info("Saved files to %s", self._serialization_dir)

    def _restore_checkpoint(self):
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from  model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        Returns
        -------
        epoch
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
        task_state_path = os.path.join(self._serialization_dir,
                                       "task_state_epoch_{}.th".format(epoch_to_load))
        metric_state_path = os.path.join(self._serialization_dir,
                                         "metric_state_epoch_{}.th".format(epoch_to_load))

        model_state = torch.load(model_path, map_location=device_mapping(self._cuda_device))
        self._model.load_state_dict(model_state)

        task_states = torch.load(task_state_path)
        for task_name, task_state in task_states.items():
            if task_name == 'global':
                continue
            self._task_infos[task_name]['total_batches_trained'] = task_state['total_batches_trained']
            self._task_infos[task_name]['optimizer'].load_state_dict(task_state['optimizer'])
            for param, val in task_state['scheduler'].items():
                setattr(self._task_infos[task_name]['scheduler'], param, val)
            self._task_infos[task_name]['stopped'] = task_state['stopped']
            generator = self._task_infos[task_name]['tr_generator']
            for _ in itertools.islice(generator, task_state['total_batches_trained']):
                pass
        if task_states['global']['optimizer'] is not None:
            self._g_optimizer.load_state_dict(task_states['global']['optimizer'])
        if task_states['global']['scheduler'] is not None:
            for param, val in task_states['global']['scheduler'].items():
                setattr(self._g_scheduler, param, val)

        metric_states = torch.load(metric_state_path)
        for metric_name, metric_state in metric_states.items():
            self._metric_infos[metric_name]['hist'] = metric_state['hist']
            self._metric_infos[metric_name]['stopped'] = metric_state['stopped']
            self._metric_infos[metric_name]['best'] = metric_state['best']

        training_state = torch.load(training_state_path)
        return training_state["pass"], training_state["should_stop"]

    @classmethod
    def from_params(cls, model, serialization_dir, iterator, params):
        ''' Generator trainer from parameters.  '''

        patience = params.pop("patience", 2)
        num_epochs = params.pop("num_epochs", 20)
        max_vals = params.pop("max_vals", 50)
        cuda_device = params.pop("cuda_device", -1)
        grad_norm = params.pop("grad_norm", None)
        grad_clipping = params.pop("grad_clipping", None)
        lr_decay = params.pop("lr_decay", None)
        min_lr = params.pop("min_lr", None)
        no_tqdm = params.pop("no_tqdm", False)

        params.assert_empty(cls.__name__)
        return SamplingMultiTaskTrainer(model,
                                        iterator,
                                        patience=patience,
                                        num_epochs=num_epochs,
                                        max_vals=max_vals,
                                        serialization_dir=serialization_dir,
                                        cuda_device=cuda_device,
                                        grad_norm=grad_norm,
                                        grad_clipping=grad_clipping,
                                        lr_decay=lr_decay,
                                        min_lr=min_lr,
                                        no_tqdm=no_tqdm)
