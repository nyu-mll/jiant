""" Trainer """

import os
import re
import glob
import time
import copy
import random
import logging as log
import itertools
import ipdb as pdb  # pylint: disable=unused-import

import torch
import torch.optim.lr_scheduler
from torch.nn.utils.clip_grad import clip_grad_norm_

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.iterators import BasicIterator, BucketIterator
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from utils import device_mapping


def build_trainer(args, model, max_vals):
    '''Build a trainer'''
    iterator = BasicIterator(args.batch_size)
    # iterator = BucketIterator(sorting_keys=[("sentence1", "num_tokens")],
    #                          batch_size=args.batch_size)

    if args.optimizer == 'adam':
        # AMSGrad is a flag variant of Adam, not its own object.
        opt_params = Params({'type': args.optimizer, 'lr': args.lr,
                             'weight_decay': 1e-5, 'amsgrad': True})
    else:
        opt_params = Params({'type': args.optimizer, 'lr': args.lr, 'weight_decay': 1e-5})

    if 'transformer' in args.sent_enc:
        schd_params = Params({'type': 'noam',
                              'model_size': args.d_hid,
                              'warmup_steps': 4000,
                              'factor': 1.0})
    else:
        schd_params = Params({'type': 'reduce_on_plateau',
                              'mode': 'max',
                              'factor': args.lr_decay_factor,
                              'patience': args.task_patience,
                              'threshold': args.scheduler_threshold,
                              'threshold_mode': 'abs',
                              'verbose': True})

    train_params = Params({'num_epochs': args.n_epochs, 'cuda_device': args.cuda,
                           'patience': args.patience, 'grad_norm': args.max_grad_norm,
                           'max_vals': max_vals,
                           'lr_decay': .99, 'min_lr': args.min_lr, 'no_tqdm': args.no_tqdm})
    trainer = SamplingMultiTaskTrainer.from_params(model, args.run_dir, iterator,
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
        task_sizes = [(idx, iterator.get_num_batches(task.train_data))
                      for idx, task in enumerate(tasks)]
        task_sizes.sort(key=lambda x: x[1], reverse=bool(method == 'large_to_small'))
        task_order = [task_idx for task_idx, _ in task_sizes]
    return task_order


class SamplingMultiTaskTrainer:
    def __init__(self, model, iterator, patience=2, num_epochs=20, max_vals=50,
                 serialization_dir=None, cuda_device=-1,
                 grad_norm=None, grad_clipping=None, lr_decay=None, min_lr=None,
                 no_tqdm=False):
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
            We use ``tqdm`` for log, which will print a nice progress bar that updates in place
            after every batch.  This is nice if you're running training on a local shell, but can
            cause problems with log files from, e.g., a docker image running on kubernetes.  If
            ``no_tqdm`` is ``True``, we will not use tqdm, and instead log batch statistics using
            ``log.info``, outputting a line at most every 10 seconds.
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
        self._summary_interval = 100  # num batches between log to tensorboard
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
            task_info['scheduler'] = LearningRateScheduler.from_params(
                task_info['optimizer'], copy.deepcopy(scheduler_params))
            task_info['stopped'] = False
            task_info['last_log'] = time.time()
        # Metric bookkeeping
        all_metrics = [task.val_metric for task in tasks] + ['micro_avg', 'macro_avg']
        metric_infos = {metric: {'hist': [], 'stopped': False, 'best': (-1, {})} for
                        metric in all_metrics}
        self._task_infos = task_infos
        self._metric_infos = metric_infos
        return task_infos, metric_infos

    def train(self, tasks, stop_metric,
              validation_interval, n_batches_per_pass,
              weighting_method, scaling_method,
              train_params, optimizer_params, scheduler_params,
              shared_optimizer=0, load_model=1, phase="main"):

        if weighting_method == 'uniform':
            log.info("Sampling tasks uniformly")
        elif weighting_method == 'proportional':
            log.info("Sampling tasks proportional to number of training batches")

        if scaling_method == 'max':
            # divide by # batches, multiply by max # batches
            log.info("Scaling losses to largest task")
        elif scaling_method == 'min':
            # divide by # batches, multiply by fewest # batches
            log.info("Scaling losses to the smallest task")
        elif scaling_method == 'unit':
            log.info("Dividing losses by number of training batches")

        iterator = self._iterator
        task_infos, metric_infos = self._setup_training(tasks, train_params, optimizer_params,
                                                        scheduler_params, iterator)
        if shared_optimizer:
            g_optimizer = Optimizer.from_params(train_params, copy.deepcopy(optimizer_params))
            g_scheduler = LearningRateScheduler.from_params(
                g_optimizer, copy.deepcopy(scheduler_params))
        else:
            g_optimizer, g_scheduler = None, None
        self._g_optimizer = g_optimizer
        self._g_scheduler = g_scheduler

        n_pass, should_stop = 0, False  # define these here b/c they might get overridden on load
        if self._serialization_dir is not None:  # Resume from serialization path
            if load_model and any(
                    ["model_state_" in x for x in os.listdir(self._serialization_dir)]):
                n_pass, should_stop = self._restore_checkpoint()
                log.info("Loaded model from checkpoint. Starting at pass %d.", n_pass)
            else:
                log.info("Not loading.")
                checkpoint_pattern = os.path.join(self._serialization_dir, "*_{}_*.th".format(phase))
                assert len(glob.glob(checkpoint_pattern)) == 0, \
                    "There are existing checkpoints here which will be overwritten." \
                    "Use -m or LOAD_MODEL to load the checkpoints instead." \
                    "If you don't want them, delete them or change your experimnent name."

        if self._grad_clipping is not None:  # pylint: disable=invalid-unary-operand-type
            def clip_function(grad): return grad.clamp(-self._grad_clipping, self._grad_clipping)
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

        log.info("Beginning training. Stopping metric: %s", stop_metric)
        all_tr_metrics = {}
        while not should_stop:
            self._model.train()

            # randomly select a task
            task = samples[n_pass % (validation_interval)]
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
                loss = output_dict["loss"]  # optionally scale loss
                if scaling_method == 'unit' and weighting_method == 'proportional':
                    loss /= task_info['n_tr_batches']
                elif scaling_method == 'max' and weighting_method == 'proportional':
                    loss *= (max_weight / task_info['n_tr_batches'])
                elif scaling_method == 'min' and weighting_method == 'proportional':
                    loss *= (min_weight / task_info['n_tr_batches'])
                loss.backward()
                assert not torch.isnan(loss).any()
                tr_loss += loss.data.cpu().numpy()

                # Gradient regularization and application
                if self._grad_norm:
                    clip_grad_norm_(self._model.parameters(), self._grad_norm)
                optimizer.step()
                n_pass += 1  # update per batch

            # Update training progress on that task
            task_info['n_batches_since_val'] = n_batches_since_val
            task_info['total_batches_trained'] = total_batches_trained
            task_info['loss'] = tr_loss

            # Intermediate log
            if time.time() - task_info['last_log'] > self._log_interval:
                task_metrics = task.get_metrics()
                task_metrics["%s_loss" % task.name] = tr_loss / n_batches_since_val
                description = self._description_from_metrics(task_metrics)
                log.info("Update %d: task %s, batch %d (%d): %s", n_pass,
                         task.name, n_batches_since_val, total_batches_trained, description)
                task_info['last_log'] = time.time()

            # Validation
            if n_pass % (validation_interval) == 0:
                epoch = int(n_pass / validation_interval)
                log.info("***** Pass %d / Epoch %d *****", n_pass, epoch)
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
                    log.info("%s: trained on %d batches, %.3f epochs", task.name,
                             n_batches_since_val, n_batches_since_val / task_info['n_tr_batches'])

                # Validate
                log.info("Validating...")
                all_val_metrics, should_save, new_best_macro, task_infos, metric_infos = self._validate(
                    epoch, tasks, task_infos, metric_infos, iterator, g_scheduler, periodic_save=(phase != "eval"))

                # Check stopping conditions
                should_stop, task_infos, metric_infos = self._check_stop(
                    epoch, stop_metric, tasks, task_infos, metric_infos, g_optimizer)

                # Log results
                for name, value in all_val_metrics.items():
                    log.info("Statistic: %s", name)
                    if name in all_tr_metrics:
                        log.info("\ttraining: %3f", all_tr_metrics[name])
                    log.info("\tvalidation: %3f", value)

                self._metric_infos = metric_infos
                self._task_infos = task_infos
                all_tr_metrics = {}
                samples = random.choices(tasks, weights=sample_weights, k=validation_interval)

                if should_save:
                    self._save_checkpoint(
                        {"pass": n_pass, "epoch": epoch, "should_stop": should_stop},
                        phase=phase, new_best_macro=new_best_macro)

        log.info('Stopped training after %d validation checks', n_pass / validation_interval)
        return self._aggregate_results(tasks, task_infos, metric_infos)  # , validation_interval)

    def _aggregate_results(self, tasks, task_infos, metric_infos):
        ''' Ad hoc helper function to print results after finishing training '''
        results = {}
        for task in tasks:
            task_info = task_infos[task.name]
            log.info('Trained %s for %d batches or %.3f epochs',
                     task.name, task_info['total_batches_trained'],
                     task_info['total_batches_trained'] / task_info['n_tr_batches'])
            results[task.name] = metric_infos[task.val_metric]['best'][0]  # * validation_interval
        results['micro'] = metric_infos['micro_avg']['best'][0]  # * validation_interval
        results['macro'] = metric_infos['macro_avg']['best'][0]  # * validation_interval
        log.info('***** VALIDATION RESULTS *****')
        for metric in metric_infos.keys():
            best_epoch, epoch_metrics = metric_infos[metric]['best']
            all_metrics_str = ', '.join(['%s: %.5f' % (metric, score) for
                                         metric, score in epoch_metrics.items()])
            log.info('%s, %d, %s', metric, best_epoch, all_metrics_str)
        return results

    def _validate(
            self,
            epoch,
            tasks,
            task_infos,
            metric_infos,
            iterator,
            g_scheduler,
            periodic_save=True):
        ''' Validate on all tasks and return the results and whether to save this epoch or not '''
        self._model.eval()
        all_val_metrics = {("%s_loss" % task.name): 0.0 for task in tasks}
        all_val_metrics["macro_avg"] = 0.0
        all_val_metrics["micro_avg"] = 0.0
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

                # log
                if time.time() - task_info['last_log'] > self._log_interval:
                    task_metrics = task.get_metrics()
                    task_metrics["%s_loss" %
                                 task.name] = all_val_metrics["%s_loss" %
                                                              task.name] / batch_num
                    description = self._description_from_metrics(task_metrics)
                    log.info("Batch %d/%d: %s", batch_num, n_val_batches, description)
                    task_info['last_log'] = time.time()
                if 'labels' in batch:
                    n_examples += batch['labels'].size()[0]
                elif 'targs' in batch:
                    n_examples += batch['targs']['words'].nelement()
            assert batch_num == n_val_batches

            # Get task validation metrics and store in all_val_metrics
            task_metrics = task.get_metrics(reset=True)
            for name, value in task_metrics.items():
                all_val_metrics["%s_%s" % (task.name, name)] = value
            all_val_metrics["%s_loss" % task.name] /= n_val_batches
            all_val_metrics["micro_avg"] += \
                all_val_metrics[task.val_metric] * n_examples
            all_val_metrics["macro_avg"] += \
                all_val_metrics[task.val_metric]
            n_examples_overall += n_examples

            # Reset training progress
            task_info['n_batches_since_val'] = 0
            task_info['loss'] = 0

        all_val_metrics['micro_avg'] /= n_examples_overall
        all_val_metrics['macro_avg'] /= len(tasks)

        # Track per task patience
        should_save = periodic_save  # whether to save this epoch or not.
        # Currently we save every validation in the main training runs.
        new_best_macro = False  # whether this epoch is a new best

        for task in tasks + ['micro', 'macro']:
            if task in ['micro', 'macro']:
                metric = "%s_avg" % task
                metric_decreases = False
            else:
                metric = task.val_metric
                metric_decreases = task.val_metric_decreases
                task = task.name
            if metric_infos[metric]['stopped']:
                continue
            this_epoch_metric = all_val_metrics[metric]
            metric_history = metric_infos[metric]['hist']
            metric_history.append(this_epoch_metric)
            is_best_so_far, out_of_patience = \
                self._check_history(metric_history, this_epoch_metric, metric_decreases)
            if is_best_so_far:
                log.info("Best model found for %s.", task)
                metric_infos[metric]['best'] = (epoch, all_val_metrics)
                should_save = True
                if task == 'macro':
                    new_best_macro = True
            if out_of_patience:
                if periodic_save:
                    should_save = True
                metric_infos[metric]['stopped'] = True
                log.info("Out of patience. Stopped tracking %s", task)

            if hasattr(task, 'name') and g_scheduler is None:  # might be "is not None"?
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

        return all_val_metrics, should_save, new_best_macro, task_infos, metric_infos

    def _check_stop(self, epoch, stop_metric, tasks, task_infos, metric_infos, g_optimizer):
        ''' Check to see if should stop '''
        if g_optimizer is None:
            stop_tr = True
            for task in tasks:
                task_info = task_infos[task.name]
                if task_info['optimizer'].param_groups[0]['lr'] < self._min_lr:
                    log.info("Minimum lr hit on %s.", task.name)
                    task_info['stopped'] = True
                stop_tr = stop_tr and task_info['stopped']
                #stop_val = stop_val and metric_infos[task.val_metric]['stopped']
        else:
            if g_optimizer.param_groups[0]['lr'] < self._min_lr:
                log.info("Minimum lr hit.")
                stop_tr = True
            else:
                stop_tr = False

        stop_val = metric_infos[stop_metric]['stopped']

        should_stop = False
        if stop_tr:
            should_stop = True
            log.info("All tasks hit minimum lr. Stopping training.")
        if stop_val:
            should_stop = True
            log.info("All metrics ran out of patience. Stopping training.")
        if epoch >= self._max_vals:
            log.info("Maximum number of validations hit. Stopping training.")
            should_stop = True

        return should_stop, task_infos, metric_infos

    def _forward(self, batch, for_training, task=None):
        # tensor_batch = arrays_to_variables(batch, self._cuda_device, for_training=for_training)
        tensor_batch = batch
        return self._model.forward(task, tensor_batch)  # , **tensor_batch)

    def _description_from_metrics(self, metrics):
        # pylint: disable=no-self-use
        return ', '.join(["%s: %.4f" % (name, value) for name, value in metrics.items()]) + " ||"

    def _unmark_previous_best(self, phase="main"):
        marked_best = glob.glob(
            os.path.join(self._serialization_dir, "*_state_{}_epoch_*.best_macro.th".format(phase)))
        for file in marked_best:
            print(file)
            os.rename(file, re.sub('%s$' % ".best_macro.th", ".th", file))

    def _save_checkpoint(self, training_state, phase="main", new_best_macro=False):
        """
        Parameters
        ----------
        epoch , required.
            The epoch of training.
        is_best, optional (default = None)
            A flag which causes the model weights at the given epoch to
            be copied to a "best.th" file. The value of this flag should
            be based on some validation metric computed by your model.
        TODO: Is there a reason this was removed?
        """
        if not self._serialization_dir:
            raise ConfigurationError("serialization_dir not specified - cannot "
                                     "restore a model without a directory path.")

        epoch = training_state["epoch"]
        if phase == "eval":
            model_path = os.path.join(
                self._serialization_dir,
                "model_state_eval_best.th")
        else:
            if new_best_macro:
                self._unmark_previous_best(phase)
                best_str = ".best_macro"
            else:
                best_str = ""

            model_path = os.path.join(
                self._serialization_dir,
                "model_state_{}_epoch_{}{}.th".format(
                    phase, epoch, best_str))

        model_state = self._model.state_dict()

        # Don't save embeddings here.
        # TODO: There has to be a prettier way to do this.
        keys_to_skip = []
        for key in model_state:
            if 'token_embedder_words' in key:
                keys_to_skip.append(key)
        for key in keys_to_skip:
            del model_state[key]

        torch.save(model_state, model_path)

        if phase != "eval":
            torch.save(
                training_state,
                os.path.join(
                    self._serialization_dir,
                    "training_state_{}_epoch_{}{}.th".format(
                        phase, epoch, best_str)))

            task_states = {}
            for task_name, task_info in self._task_infos.items():
                task_states[task_name] = {}
                task_states[task_name]['total_batches_trained'] = task_info['total_batches_trained']
                task_states[task_name]['stopped'] = task_info['stopped']
                task_states[task_name]['optimizer'] = task_info['optimizer'].state_dict()
                sched = task_info['scheduler']
                sched_params = {}  # {'best': sched.best, 'num_bad_epochs': sched.num_bad_epochs,
                #'cooldown_counter': sched.cooldown_counter}
                task_states[task_name]['scheduler'] = sched_params
            task_states['global'] = {}
            task_states['global']['optimizer'] = self._g_optimizer.state_dict() if \
                self._g_optimizer is not None else None
            if self._g_scheduler is not None:
                sched = self._g_scheduler
                sched_params = {}  # {'best': sched.best, 'num_bad_epochs': sched.num_bad_epochs,
                #'cooldown_counter': sched.cooldown_counter}
                task_states['global']['scheduler'] = sched_params
            else:
                task_states['global']['scheduler'] = None
            torch.save(task_states, os.path.join(self._serialization_dir,
                                                 "task_state_{}_epoch_{}{}.th".format(
                                                     phase, epoch, best_str)))

            metric_states = {}
            for metric_name, metric_info in self._metric_infos.items():
                metric_states[metric_name] = {}
                metric_states[metric_name]['hist'] = metric_info['hist']
                metric_states[metric_name]['stopped'] = metric_info['stopped']
                metric_states[metric_name]['best'] = metric_info['best']
            torch.save(
                metric_states,
                os.path.join(
                    self._serialization_dir,
                    "metric_state_{}_epoch_{}{}.th".format(
                        phase, epoch, best_str)))
        log.info("Saved files to %s", self._serialization_dir)

    def _find_last_checkpoint_suffix(self, search_phases_in_priority_order=['main']):
        """
        Search for checkpoints to load, looking only for `main` training checkpoints.

        TODO: This is probably hairier than it needs to be. If you're good at string handling...
        """
        if not self._serialization_dir:
            raise ConfigurationError("serialization_dir not specified - cannot "
                                     "restore a model without a directory path.")

        for current_search_phase in search_phases_in_priority_order:
            max_epoch = 0
            to_return = None
            candidate_files = glob.glob(
                os.path.join(
                    self._serialization_dir,
                    "model_state_{}_*".format(current_search_phase)))
            for x in candidate_files:
                epoch = int(x.split("model_state_{}_epoch_".format(
                    current_search_phase))[-1].split(".")[0])
                if epoch >= max_epoch:
                    max_epoch = epoch
                    to_return = x
            return to_return.split("model_state_")[-1]

    def _restore_checkpoint(self, search_phases_in_priority_order=['main']):
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

        suffix_to_load = self._find_last_checkpoint_suffix(
            search_phases_in_priority_order=search_phases_in_priority_order)
        assert suffix_to_load, "No checkpoint found."
        log.info("Found checkpoint {}. Loading.".format(suffix_to_load))

        model_path = os.path.join(self._serialization_dir,
                                  "model_state_{}".format(suffix_to_load))
        training_state_path = os.path.join(self._serialization_dir,
                                           "training_state_{}".format(suffix_to_load))
        task_state_path = os.path.join(self._serialization_dir,
                                       "task_state_{}".format(suffix_to_load))
        metric_state_path = os.path.join(self._serialization_dir,
                                         "metric_state_{}".format(suffix_to_load))

        model_state = torch.load(model_path, map_location=device_mapping(self._cuda_device))
        self._model.load_state_dict(model_state, strict=False)

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
            for _ in itertools.islice(generator, task_state['total_batches_trained'] %
                                      self._task_infos[task_name]['n_tr_batches']):
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
        return SamplingMultiTaskTrainer(model, iterator, patience=patience,
                                        num_epochs=num_epochs, max_vals=max_vals,
                                        serialization_dir=serialization_dir,
                                        cuda_device=cuda_device, grad_norm=grad_norm,
                                        grad_clipping=grad_clipping, lr_decay=lr_decay,
                                        min_lr=min_lr, no_tqdm=no_tqdm)
