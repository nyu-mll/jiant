from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List

import torch
import torch.nn as nn
import transformers


import jiant.proj.main.components.container_setup as container_setup
import jiant.proj.main.modeling.primary as primary
import jiant.utils.python.strings as strings

from jiant.proj.main.modeling.heads import JiantHeadFactory
from jiant.proj.main.modeling.taskmodels import JiantTaskModelFactory, Taskmodel, MLMModel

from jiant.shared.model_resolution import ModelArchitectures
from jiant.tasks import Task


def setup_jiant_model(
    hf_pretrained_model_name_or_path: str,
    model_config_path: str,
    task_dict: Dict[str, Task],
    taskmodels_config: container_setup.TaskmodelsConfig,
):
    """Sets up tokenizer, encoder, and task models, and instantiates and returns a JiantModel.

    Args:
        hf_pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
            Can be either:

                - A string, the `model id` of a predefined tokenizer hosted inside a model
                  repo on huggingface.co. Valid model ids can be located at the root-level,
                  like ``bert-base-uncased``, or namespaced under
                  a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - A path to a `directory` containing vocabulary files required by the
                  tokenizer, for instance saved using the
                  :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.,
                  ``./my_model_directory/``.
                - A path or url to a single saved vocabulary file if and only if
                  the tokenizer only requires a single vocabulary file (like Bert or XLNet),
                  e.g.: ``./my_model_directory/vocab.txt``. (Not
                  applicable to all derived classes)
        model_config_path (str): Path to the JSON file containing the configuration parameters.
        task_dict (Dict[str, tasks.Task]): map from task name to task instance.
        taskmodels_config: maps mapping from tasks to models, and specifying task-model configs.

    Returns:
        JiantModel nn.Module.

    """
    hf_model = transformers.AutoModel.from_pretrained(hf_pretrained_model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_pretrained_model_name_or_path)
    jiant_transformers_model = primary.JiantTransformersModelFactory()(hf_model)
    taskmodels_dict = {
        taskmodel_name: create_taskmodel(
            task=task_dict[task_name_list[0]],  # Take the first task
            jiant_transformers_model=jiant_transformers_model,
            taskmodel_kwargs=taskmodels_config.get_taskmodel_kwargs(taskmodel_name),
        )
        for taskmodel_name, task_name_list in get_taskmodel_and_task_names(
            taskmodels_config.task_to_taskmodel_map
        ).items()
    }
    return primary.JiantModel(
        task_dict=task_dict,
        encoder=jiant_transformers_model,
        taskmodels_dict=taskmodels_dict,
        task_to_taskmodel_map=taskmodels_config.task_to_taskmodel_map,
        tokenizer=tokenizer,
    )


def delegate_load_from_path(jiant_model: primary.JiantModel, weights_path: str, load_mode: str):
    """Load weights dict from file and load weights according to specified loading mode.

    Args:
        jiant_model (JiantModel): jiant model (encoder and task models are core components).
        weights_path (str): filepath to weights object saved with torch.save().
        load_mode (str): TODO

    Returns:
        TODO: return behavior is not consistent between load_mode options, clarify as needed here.

    """
    weights_dict = torch.load(weights_path)
    return delegate_load(jiant_model=jiant_model, weights_dict=weights_dict, load_mode=load_mode)


def delegate_load(jiant_model, weights_dict: dict, load_mode: str):
    """Load weights dict into JiantModel according to specified loading mode.

    Args:
        jiant_model (JiantModel): jiant model (encoder and task models are core components).
        weights_dict (Dict): model weights.
        load_mode: TODO

    Returns:
        TODO: return behavior is not consistent between load_mode options, clarify as needed here.

    """
    if load_mode == "from_transformers":
        return load_encoder_from_transformers_weights(
            encoder=jiant_model.encoder, weights_dict=weights_dict,
        )
    elif load_mode == "from_transformers_with_mlm":
        remainder = load_encoder_from_transformers_weights(
            encoder=jiant_model.encoder, weights_dict=weights_dict, return_remainder=True,
        )
        load_lm_heads_from_transformers_weights(
            jiant_model=jiant_model, weights_dict=remainder,
        )
        return
    elif load_mode == "all":
        jiant_model.load_state_dict(weights_dict)
    elif load_mode == "encoder_only":
        return load_encoder_only(jiant_model=jiant_model, weights_dict=weights_dict)
    elif load_mode == "partial_weights":
        return load_partial_heads(
            jiant_model=jiant_model, weights_dict=weights_dict, allow_missing_head_weights=True,
        )
    elif load_mode == "partial_heads":
        return load_partial_heads(
            jiant_model=jiant_model, weights_dict=weights_dict, allow_missing_head_model=True,
        )
    elif load_mode == "partial":
        return load_partial_heads(
            jiant_model=jiant_model,
            weights_dict=weights_dict,
            allow_missing_head_weights=True,
            allow_missing_head_model=True,
        )
    else:
        raise KeyError(load_mode)


def load_encoder_from_transformers_weights(
    encoder: nn.Module, weights_dict: dict, return_remainder=False
):
    """Find encoder weights in weights dict, load them into encoder, return any remaining weights.

    TODO: clarify how we know the encoder weights will be prefixed by transformer name.

    Args:
        encoder (PreTrainedModel): Transformer w/o heads (embedding layer + self-attention layer).
        weights_dict (Dict): model weights.
        return_remainder (bool): If True, return any leftover weights.

    Returns:
        Dict containing any leftover weights.

    """
    remainder_weights_dict = {}
    load_weights_dict = {}
    model_arch = ModelArchitectures.from_model_type(model_type=encoder.config.model_type)
    encoder_prefix = model_arch.value + "."
    # Encoder
    for k, v in weights_dict.items():
        if k.startswith(encoder_prefix):
            load_weights_dict[strings.remove_prefix(k, encoder_prefix)] = v
        else:
            remainder_weights_dict[k] = v
    encoder.load_state_dict(load_weights_dict)
    if return_remainder:
        return remainder_weights_dict


def load_lm_heads_from_transformers_weights(jiant_model, weights_dict):
    mlm_weights_dict = jiant_model.encoder.get_mlm_weights_dict(weights_dict)
    missed = set()
    for taskmodel_name, taskmodel in jiant_model.taskmodels_dict.items():
        if not isinstance(taskmodel, MLMModel):
            continue
        mismatch = taskmodel.mlm_head.load_state_dict(mlm_weights_dict)
        assert not mismatch.missing_keys
        missed.update(mismatch.unexpected_keys)
        taskmodel.mlm_head.decoder.weight = jiant_model.encoder.embeddings.word_embeddings.weight
    return list(missed)


def load_encoder_only(jiant_model, weights_dict):
    """Loads only encoder weights

    Args:
        jiant_model (JiantModel): jiant model (encoder and task models are core components).
        weights_dict (Dict): model weights.

    Returns:
        Dict[str, List] containing dropped keys

    """
    new_weights_dict = {}
    encoder_keys = [n for n, p in jiant_model.encoder.named_parameters()]

    # 1. Handle core encoder
    for encoder_key in encoder_keys:
        new_key = f"encoder.{encoder_key}"
        new_weights_dict[new_key] = weights_dict[new_key]

    # 2. Handle taskmodel encoders:
    for taskmodel_key in jiant_model.task_to_taskmodel_map:
        for encoder_key in encoder_keys:
            new_key = f"taskmodels_dict.{taskmodel_key}.encoder.{encoder_key}"
            new_weights_dict[new_key] = weights_dict[f"encoder.{encoder_key}"]

    mismatch = jiant_model.load_state_dict(new_weights_dict, strict=False)
    assert not mismatch.unexpected_keys
    return {
        "dropped_keys": mismatch.missing_keys,
    }


def load_partial_heads(
    jiant_model, weights_dict, allow_missing_head_weights=False, allow_missing_head_model=False
):
    """Loads model weights and returns lists of missing head weights or missing heads (if any).

    Args:
        jiant_model (JiantModel): jiant model (encoder and task models are core components).
        weights_dict (Dict): model weights.
        allow_missing_head_weights (bool): If False, throw exception if there are missing keys.
        allow_missing_head_model (bool): If False, throw exception if there are unexpected keys.

    Returns:
        Dict[str, List] containing lists of missing head weights or missing heads if any.

    """
    mismatch = jiant_model.load_state_dict(weights_dict, strict=False)
    result = {}
    if mismatch.missing_keys:
        assert allow_missing_head_weights
        missing_head_weights = set()
        for k in mismatch.missing_keys:
            missing_head_weights.add(k.split(".")[1])
        result["missing_head_weights"] = list(missing_head_weights)
    if mismatch.unexpected_keys:
        assert allow_missing_head_model
        missing_heads_model = set()
        for k in mismatch.unexpected_keys:
            missing_heads_model.add(k.split(".")[1])
        result["missing_heads_model"] = list(missing_heads_model)
    return result


def create_taskmodel(task, jiant_transformers_model, **taskmodel_kwargs) -> Taskmodel:
    """Creates, initializes and returns the task model for a given task type and encoder.

    Args:
        task (Task): Task object associated with the taskmodel being created.
        jiant_transformers_model (JiantTransformersModel): Transformer w/o heads
            (embedding layer + self-attention layer).
        **taskmodel_kwargs: Additional args for taskmodel setup

    Raises:
        KeyError if task does not have valid TASK_TYPE.

    Returns:
        Taskmodel

    """
    head_kwargs = {}
    head_kwargs["hidden_size"] = jiant_transformers_model.get_hidden_size()
    head_kwargs["hidden_dropout_prob"] = jiant_transformers_model.get_hidden_dropout_prob()
    head_kwargs["vocab_size"] = jiant_transformers_model.config.vocab_size
    head_kwargs["model_arch"] = ModelArchitectures(jiant_transformers_model.config.model_type)

    if hasattr(jiant_transformers_model, "hidden_act"):
        head_kwargs["hidden_act"] = jiant_transformers_model.config.hidden_act
    if hasattr(jiant_transformers_model, "layer_norm_eps"):
        head_kwargs["layer_norm_eps"] = jiant_transformers_model.config.layer_norm_eps

    head = JiantHeadFactory()(task, **head_kwargs)

    taskmodel = JiantTaskModelFactory()(task, jiant_transformers_model, head, **taskmodel_kwargs)
    return taskmodel


@dataclass
class TransformersClassSpec:
    config_class: Any
    tokenizer_class: Any
    model_class: Any


def get_taskmodel_and_task_names(task_to_taskmodel_map: Dict[str, str]) -> Dict[str, List[str]]:
    """Get mapping from task model name to the list of task names associated with that task model.

    Args:
        task_to_taskmodel_map (Dict[str, str]): map from task name to task model name.

    Returns:
        Dict[str, List[str]] map of task model names to lists of task names using that task model.

    """
    taskmodel_and_task_names = {}
    for task_name, taskmodel_name in task_to_taskmodel_map.items():
        if taskmodel_name not in taskmodel_and_task_names:
            taskmodel_and_task_names[taskmodel_name] = []
        taskmodel_and_task_names[taskmodel_name].append(task_name)
    return taskmodel_and_task_names


def get_ancestor_model(transformers_class_spec, model_config_path):
    """Load the model config from a file, configure the model, and return the model.

    This function returns the model class with all the pretrained weights. E.g., for BERT this is
    BertForPreTraining which includes masked language modeling and next sentence prediction heads.

    Args:
        transformers_class_spec (TransformersClassSpec): has refs to model, tokenizer, and config.
        model_config_path (str): Path to the JSON file containing the configuration parameters.

    Returns:
        Configured model.

    """
    config = transformers_class_spec.config_class.from_json_file(model_config_path)
    model = transformers_class_spec.model_class(config)
    return model
