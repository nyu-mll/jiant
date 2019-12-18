"""
Assorted utilities for working with neural networks in AllenNLP.
"""
import codecs
import copy
import json
import logging
import os
from pkg_resources import resource_filename
import random
import time
from typing import Dict, Iterable, List, Optional, Sequence, Union
import glob
import numpy as np
import torch
import jsondiff

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.util import device_mapping, masked_softmax
from sacremoses import MosesDetokenizer
from torch.autograd import Variable
from torch.nn import Dropout, Linear, Parameter, init

from .config import Params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

SOS_TOK, EOS_TOK = "<SOS>", "<EOS>"

# Note: using the full 'detokenize()' method is not recommended, since it does
# a poor job of adding correct whitespace. Use unescape_xml() only.
_MOSES_DETOKENIZER = MosesDetokenizer()


def get_output_attribute(out, attribute_name, cuda_device):
    """
    This function handles processing/reduction of output for both
    DataParallel or non-DataParallel situations.
    For the case of multiple GPUs, This function will
    sum all values for a certain output attribute in various batches
    together.

    Parameters
    ---------------------
    out: Dictionary, output of model during forward pass,
    attribute_name: str,
    cuda_device: list or int
    """
    if isinstance(cuda_device, list):
        return out[attribute_name].sum()
    else:
        return out[attribute_name]


def get_model_attribute(model, attribute_name, cuda_device):
    """
        Getter function for both CPU and GPU.

        Parameters
        ____________________
        model: MultiTaskModel object,
        attribute_name: str

        Returns
        --------------------
        The attribute object from the model.
    """
    # maybe we should do (int, list)
    if isinstance(cuda_device, list):
        return getattr(model.module, attribute_name)
    else:
        return getattr(model, attribute_name)


def select_pool_type(args):
    """
        Select a sane default sequence pooling type.
    """
    if args.pool_type == "auto":
        pool_type = "max"
        if args.sent_enc == "none":
            if (
                args.input_module.startswith("bert-")
                or args.input_module.startswith("roberta-")
                or args.input_module.startswith("xlm-")
            ):
                pool_type = "first"
            elif (
                args.input_module.startswith("xlnet-")
                or args.input_module.startswith("openai-gpt")
                or args.input_module.startswith("gpt2")
                or args.input_module.startswith("transfo-xl-")
            ):
                pool_type = "final"
    else:
        pool_type = args.pool_type
    return pool_type


def apply_standard_boundary_tokens(s1, s2=None):
    """Apply <SOS> and <EOS> to sequences of string-valued tokens.
    Corresponds to more complex functions used with models like XLNet and BERT.
    """
    assert not s2, "apply_standard_boundary_tokens only supports single sequences"
    return [SOS_TOK] + s1 + [EOS_TOK]


def check_for_previous_checkpoints(serialization_dir, tasks, phase, load_model):
    """
    Check if there are previous checkpoints.
    If phase == target_train, we loop through each of the tasks from last to first
    to find the task with the most recent checkpoint.
    If phase == pretrain, we check if there is a most recent checkpoint in the run
    directory.

    Parameters
    ---------------------
    serialization_dir: str,
    tasks: List of SamplingMultiTask objects,
    phase: str
    load_model: bool

    Returns
    ---------------------
    ckpt_directory: None or str, name of directory that checkpoints are in
    with regards to the run directory.
    val_pass: int, -1 if not found.
    suffix: None or str, the suffix of the checkpoint.
    """
    ckpt_directory = None
    ckpt_epoch = -1
    ckpt_suffix = None
    if phase == "target_train":
        for task in tasks[::-1]:
            val_pass, suffix = find_last_checkpoint_epoch(serialization_dir, phase, task.name)
            # If we have found a task with a valid checkpoint for the first time.
            if val_pass > -1 and ckpt_directory is None and phase == "target_train":
                ckpt_directory = task.name
                ckpt_epoch = val_pass
                ckpt_suffix = suffix
    else:
        ckpt_epoch, ckpt_suffix = find_last_checkpoint_epoch(serialization_dir, phase, "")
        if ckpt_epoch > -1:
            # If there exists a pretraining checkpoint, set ckpt_directory.
            ckpt_directory = ""
    if ckpt_directory is not None:
        assert_for_log(
            load_model,
            "There are existing checkpoints in %s which will be overwritten. "
            "If you are restoring from a run, or would like to train from an "
            "existing checkpoint, Use load_model = 1 to load the checkpoints instead. "
            "If you don't want them, delete them or change your experiment name."
            % serialization_dir,
        )
    return ckpt_directory, ckpt_epoch, ckpt_suffix


def find_last_checkpoint_epoch(serialization_dir, search_phase="pretrain", task_name=""):

    """
    Search for the last epoch in a directory.
    Here, we check that all four checkpoints (model, training_state, task_state, metrics)
    exist and return the most recent epoch with all four checkpoints.

    """
    if not serialization_dir:
        raise ConfigurationError(
            "serialization_dir not specified - cannot restore a model without a directory path."
        )
    suffix = None
    max_val_pass = -1
    candidate_files = glob.glob(
        os.path.join(serialization_dir, task_name, "*_state_{}_*".format(search_phase))
    )
    val_pass_to_files = {}
    for x in candidate_files:
        val_pass = int(x.split("_state_{}_val_".format(search_phase))[-1].split(".")[0])
        if not val_pass_to_files.get(val_pass):
            val_pass_to_files[val_pass] = 0
        val_pass_to_files[val_pass] += 1
        if val_pass >= max_val_pass and val_pass_to_files[val_pass] == 4:
            max_val_pass = val_pass
            suffix = x
    if suffix is not None:
        suffix = suffix.split(serialization_dir)[-1]
        suffix = "_".join(suffix.split("_")[1:])
    return max_val_pass, suffix


def copy_iter(elems):
    """Simple iterator yielding copies of elements."""
    for elem in elems:
        yield copy.deepcopy(elem)


def wrap_singleton_string(item: Union[Sequence, str]):
    """ Wrap a single string as a list. """
    if isinstance(item, str):
        # Can't check if iterable, because a string is an iterable of
        # characters, which is not what we want.
        return [item]
    return item


def sort_param_recursive(data):
    """
    Sorts the keys of a config.Params object in a param object recursively.
    """
    import pyhocon

    if isinstance(data, dict) and not isinstance(data, pyhocon.ConfigTree):
        for name, _ in list(data.items()):
            data[name] = sort_param_recursive(data[name])
    else:
        if isinstance(data, pyhocon.ConfigTree):
            data = dict(sorted(data.items(), key=lambda x: x[0]))
    return data


def parse_json_diff(diff):
    """
    Parses the output of jsondiff's diff() function, which introduces
    symbols such as replace.
    The potential keys introduced are jsondiff.replace, jsondiff.insert, and jsondiff.delete.
    For jsondiff.replace and jsondiff.insert, we simply want to return the
    actual value of the replaced or inserted item, whereas for jsondiff.delete, we do not want to
    show deletions in our parameters.
    For example, for jsondiff.replace, the output of jsondiff may be the below:
    {'mrpc': {replace: ConfigTree([('classifier_dropout', 0.1), ('classifier_hid_dim', 256),
                                   ('max_vals', 8), ('val_interval', 1)])}}
    since 'mrpc' was overriden in demo.conf. Thus, we only want to show the update and delete
    the replace. The output of this function will be:
    {'mrpc': ConfigTree([('classifier_dropout', 0.1), ('classifier_hid_dim', 256),
                         ('max_vals', 8), ('val_interval', 1)])}
    See for more information on jsondiff.
    """
    new_diff = {}
    if isinstance(diff, dict):
        for name, value in list(diff.items()):

            if name == jsondiff.replace or name == jsondiff.insert:
                # get rid of the added jsondiff key
                return value

            if name == jsondiff.delete:
                del diff[name]
                return None

            output = parse_json_diff(diff[name])
            if output:
                diff[name] = output
    return diff


def select_relevant_print_args(args):
    """
        Selects relevant arguments to print out.
        We select relevant arguments as the difference between defaults.conf and the experiment's
        configuration.

        Params
        -----------
        args: Params object

        Returns
        -----------
        return_args: Params object with only relevant arguments
        """
    import pyhocon
    from pathlib import Path

    exp_config_file = os.path.join(args.run_dir, "params.conf")
    root_directory = Path(__file__).parents[2]
    defaults_file = resource_filename("jiant", "/config/defaults.conf")
    exp_basedir = os.path.dirname(exp_config_file)
    default_basedir = os.path.dirname(defaults_file)
    fd = open(exp_config_file, "r")
    exp_config_string = fd.read()
    exp_config_string += "\n"
    fd = open(defaults_file, "r")
    default_config_string = fd.read()
    default_config_string += "\n"
    exp_config = dict(
        pyhocon.ConfigFactory.parse_string(exp_config_string, basedir=exp_basedir).items()
    )
    default_config = dict(
        pyhocon.ConfigFactory.parse_string(default_config_string, basedir=default_basedir).items()
    )
    sorted_exp_config = sort_param_recursive(exp_config)
    sorted_defaults_config = sort_param_recursive(default_config)
    diff_args = parse_json_diff(jsondiff.diff(sorted_defaults_config, sorted_exp_config))
    diff_args = Params.clone(diff_args)
    result_args = select_task_specific_args(args, diff_args)
    return result_args


def select_task_specific_args(exp_args, diff_args):
    """
    A helper function that adds in task-specific parameters from the experiment
    configurations for tasks in pretrain_tasks and target_tasks.
    """
    exp_tasks = []
    if diff_args.get("pretrain_tasks"):
        exp_tasks = diff_args.pretrain_tasks.split(",")
    if diff_args.get("target_tasks"):
        exp_tasks += diff_args.target_tasks.split(",")
    if len(exp_tasks) == 0:
        return diff_args
    for key, value in list(exp_args.as_dict().items()):
        stripped_key = key.replace("_", " ")
        stripped_key = stripped_key.replace("-", " ")
        param_task = None
        # For each parameter, identify the task the parameter relates to (if any)
        for task in exp_tasks:
            if task in stripped_key and (("edges" in stripped_key) == ("edges" in task)):
                # special logic for edges since there are edge versions of various
                # tasks.
                param_task = task
        # Add parameters that pertain to the experiment tasks
        if param_task and param_task in exp_tasks:
            diff_args[key] = value
    return diff_args


def load_model_state(model, state_path, gpu_id, skip_task_models=[], strict=True):
    """ Helper function to load a model state

    Parameters
    ----------
    model: The model object to populate with loaded parameters.
    state_path: The path to a model_state checkpoint.
    gpu_id: The GPU to use. -1 for no GPU.
    skip_task_models: If set, skip task-specific parameters for these tasks.
        This does not necessarily skip loading ELMo scalar weights, but I (Sam) sincerely
        doubt that this matters.
    strict: Whether we should fail if any parameters aren't found in the checkpoint. If false,
        there is a risk of leaving some parameters in their randomly initialized state.
    """
    model_state = torch.load(state_path)

    assert_for_log(
        not (skip_task_models and strict),
        "Can't skip task models while also strictly loading task models. Something is wrong.",
    )

    for name, param in model.named_parameters():
        # Make sure no trainable params are missing.
        if param.requires_grad:
            if strict:
                assert_for_log(
                    name in model_state,
                    "In strict mode and failed to find at least one parameter: " + name,
                )
            elif (name not in model_state) and ((not skip_task_models) or ("_mdl" not in name)):
                logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                logging.error("Parameter missing from checkpoint: " + name)
                logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    if skip_task_models:
        keys_to_skip = []
        for task in skip_task_models:
            new_keys_to_skip = [key for key in model_state if "%s_mdl" % task in key]
            if new_keys_to_skip:
                logging.info("Not loading task-specific parameters for task: %s" % task)
                keys_to_skip += new_keys_to_skip
            else:
                logging.info("Found no task-specific parameters to skip for task: %s" % task)
        for key in keys_to_skip:
            del model_state[key]

    model.load_state_dict(model_state, strict=False)
    logging.info("Loaded model state from %s", state_path)


def get_elmo_mixing_weights(text_field_embedder, task=None):
    """ Get pre-softmaxed mixing weights for ELMo from text_field_embedder for a given task.
    Stops program execution if something goes wrong (e.g. task is malformed,
    resulting in KeyError).

    args:
        - text_field_embedder (ElmoTextFieldEmbedder): the embedder used during the run
        - task (Task): a Task object with a populated `_classifier_name` attribute.

    returns:
        Dict[str, float]: dictionary with the values of each layer weight and of the scaling
                          factor.
    """
    elmo = text_field_embedder.token_embedder_elmo._elmo
    if task:
        task_id = text_field_embedder.task_map[task._classifier_name]
    else:
        task_id = text_field_embedder.task_map["@pretrain@"]
    task_weights = getattr(elmo, "scalar_mix_%d" % task_id)
    params = {
        "layer%d" % layer_id: p.item()
        for layer_id, p in enumerate(task_weights.scalar_parameters.parameters())
    }
    params["gamma"] = task_weights.gamma
    return params


def format_output(obj, cuda_devices):
    """
    Format output based on whether model is using DataParallel or not.
    DataParallel necessitates objects to be gathered into GPU:0 to have
    dimension 0.
    This function will be used for scalar outputs of model forwards
    such as loss and n_exs.
    """
    if isinstance(cuda_devices, list):
        if not isinstance(obj, torch.Tensor):
            obj = torch.tensor(obj).cuda()
        return obj.unsqueeze(0)
    else:
        return obj


def uses_cuda(cuda_devices):
    return isinstance(cuda_devices, list) or (isinstance(cuda_devices, int) and cuda_devices >= 0)


def get_batch_size(batch, cuda_devices, keyword="input"):
    """ Given a batch with unknown text_fields, get an estimate of batch size """
    if keyword == "input":
        batch_field = batch["inputs"] if "inputs" in batch else batch["input1"]
    else:
        batch_field = batch[keyword]
    keys = [k for k in batch_field.keys()]
    batch_size = batch_field[keys[0]].size()[0]
    return format_output(batch_size, cuda_devices)


def get_batch_utilization(batch_field, pad_idx=0):
    """ Get ratio of batch elements that are padding

    Batch should be field, i.e. a dictionary of inputs"""
    if "elmo" in batch_field:
        idxs = batch_field["elmo"]
        pad_ratio = idxs.eq(pad_idx).sum().item() / idxs.nelement()
    else:
        raise NotImplementedError
    return 1 - pad_ratio


def maybe_make_dir(dirname):
    """Make a directory if it doesn't exist."""
    os.makedirs(dirname, exist_ok=True)


def unescape_moses(moses_tokens):
    """Unescape Moses punctuation tokens.
    Replaces escape sequences like &#91; with the original characters
    (such as '['), so they better align to the original text.
    """
    return [_MOSES_DETOKENIZER.unescape_xml(t) for t in moses_tokens]


def load_json_data(filename: str) -> Iterable:
    """ Load JSON records, one per line. """
    with open(filename, "r") as fd:
        for line in fd:
            yield json.loads(line)


def load_lines(filename: str) -> Iterable[str]:
    """ Load text data, yielding each line. """
    with open(filename) as fd:
        for line in fd:
            yield line.strip()


def split_data(data, ratio, shuffle=1):
    """Split dataset according to ratio, larger split is first return"""
    n_exs = len(data[0])
    split_pt = int(n_exs * ratio)
    splits = [[], []]
    for col in data:
        splits[0].append(col[:split_pt])
        splits[1].append(col[split_pt:])
    return tuple(splits[0]), tuple(splits[1])


@Seq2SeqEncoder.register("masked_multi_head_self_attention")
class MaskedMultiHeadSelfAttention(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .  # noqa

    The attention mechanism is a weighted sum of a projection V of the inputs, with respect
    to the scaled, normalised dot product of Q and K, which are also both linear projections
    of the input. This procedure is repeated for each attention head, using different parameters.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    attention_dim ``int``, required.
        The dimension of the query and key projections which comprise the
        dot product attention function.
    values_dim : ``int``, required.
        The dimension which the input is projected to for representing the values,
        which are combined using the attention.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """

    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        attention_dim: int,
        values_dim: int,
        output_projection_dim: int = None,
        attention_dropout_prob: float = 0.1,
    ) -> None:
        super(MaskedMultiHeadSelfAttention, self).__init__()

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = output_projection_dim or input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim

        self._query_projections = Parameter(torch.FloatTensor(num_heads, input_dim, attention_dim))
        self._key_projections = Parameter(torch.FloatTensor(num_heads, input_dim, attention_dim))
        self._value_projections = Parameter(torch.FloatTensor(num_heads, input_dim, values_dim))

        self._scale = input_dim ** 0.5
        self._output_projection = Linear(num_heads * values_dim, self._output_dim)
        self._attention_dropout = Dropout(attention_dropout_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Because we are doing so many torch.bmm calls, which is fast but unstable,
        # it is critically important to initialise the parameters correctly such
        # that these matrix multiplications are well conditioned initially.
        # Without this initialisation, this (non-deterministically) produces
        # NaNs and overflows.
        init.xavier_normal_(self._query_projections)
        init.xavier_normal_(self._key_projections)
        init.xavier_normal_(self._value_projections)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def forward(
        self,  # pylint: disable=arguments-differ
        inputs: torch.Tensor,
        mask: torch.LongTensor = None,
    ) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        num_heads = self._num_heads

        batch_size, timesteps, hidden_dim = inputs.size()
        if mask is None:
            mask = Variable(inputs.data.new(batch_size, timesteps).fill_(1.0))

        # Treat the queries, keys and values each as a ``num_heads`` size batch.
        # shape (num_heads, batch_size * timesteps, hidden_dim)
        inputs_per_head = inputs.repeat(num_heads, 1, 1).view(
            num_heads, batch_size * timesteps, hidden_dim
        )
        # Do the projections for all the heads at once.
        # Then reshape the result as though it had a
        # (num_heads * batch_size) sized batch.
        queries_per_head = torch.bmm(inputs_per_head, self._query_projections)
        # shape (num_heads * batch_size, timesteps, attention_dim)
        queries_per_head = queries_per_head.view(
            num_heads * batch_size, timesteps, self._attention_dim
        )

        keys_per_head = torch.bmm(inputs_per_head, self._key_projections)
        # shape (num_heads * batch_size, timesteps, attention_dim)
        keys_per_head = keys_per_head.view(num_heads * batch_size, timesteps, self._attention_dim)

        values_per_head = torch.bmm(inputs_per_head, self._value_projections)
        # shape (num_heads * batch_size, timesteps, attention_dim)
        values_per_head = values_per_head.view(num_heads * batch_size, timesteps, self._values_dim)

        # shape (num_heads * batch_size, timesteps, timesteps)
        scaled_similarities = (
            torch.bmm(queries_per_head, keys_per_head.transpose(1, 2)) / self._scale
        )

        # Masking should go here
        causality_mask = subsequent_mask(timesteps).cuda()
        masked_scaled_similarities = scaled_similarities.masked_fill(causality_mask == 0, -1e9)

        # shape (num_heads * batch_size, timesteps, timesteps)
        # Normalise the distributions, using the same mask for all heads.
        attention = masked_softmax(masked_scaled_similarities, mask.repeat(num_heads, 1))
        attention = self._attention_dropout(attention)
        # This is doing the following batch-wise matrix multiplication:
        # (num_heads * batch_size, timesteps, timesteps) *
        # (num_heads * batch_size, timesteps, values_dim)
        # which is equivalent to a weighted sum of the values with respect to
        # the attention distributions for each element in the num_heads * batch_size
        # dimension.
        # shape (num_heads * batch_size, timesteps, values_dim)
        outputs = torch.bmm(attention, values_per_head)

        # Reshape back to original shape (batch_size, timesteps, num_heads * values_dim)
        # Note that we _cannot_ use a reshape here, because this tensor was created
        # with num_heads being the first dimension, so reshaping naively would not
        # throw an error, but give an incorrect result.
        outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=-1)

        # Project back to original input size.
        # shape (batch_size, timesteps, input_size)
        outputs = self._output_projection(outputs)
        return outputs

    @classmethod
    def from_params(cls, params: Params) -> "MaskedMultiHeadSelfAttention":
        num_heads = params.pop_int("num_heads")
        input_dim = params.pop_int("input_dim")
        attention_dim = params.pop_int("attention_dim")
        values_dim = params.pop_int("values_dim")
        output_projection_dim = params.pop_int("output_projection_dim", None)
        attention_dropout_prob = params.pop_float("attention_dropout_prob", 0.1)
        params.assert_empty(cls.__name__)
        return cls(
            num_heads=num_heads,
            input_dim=input_dim,
            attention_dim=attention_dim,
            values_dim=values_dim,
            output_projection_dim=output_projection_dim,
            attention_dropout_prob=attention_dropout_prob,
        )


def assert_for_log(condition, error_message):
    assert condition, error_message


def delete_all_checkpoints(serialization_dir):
    common_checkpoints = glob.glob(os.path.join(serialization_dir, "*.th"))
    task_checkpoints = glob.glob(os.path.join(serialization_dir, "*", "*.th"))
    for file in common_checkpoints + task_checkpoints:
        os.remove(file)


def transpose_list_of_lists(ls):
    if len(ls) == 0:
        return []
    return [[ls[i][j] for i in range(len(ls))] for j in range(len(ls[0]))]
