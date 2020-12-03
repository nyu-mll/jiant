from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import transformers


import jiant.proj.main.components.container_setup as container_setup
import jiant.proj.main.modeling.primary as primary
import jiant.proj.main.modeling.taskmodels as taskmodels
import jiant.proj.main.modeling.heads as heads
import jiant.shared.model_setup as model_setup
import jiant.utils.python.strings as strings
from jiant.shared.model_setup import ModelArchitectures
from jiant.tasks import Task, TaskTypes


def setup_jiant_model(
    model_type: str,
    model_config_path: str,
    tokenizer_path: str,
    task_dict: Dict[str, Task],
    taskmodels_config: container_setup.TaskmodelsConfig,
):
    """Sets up tokenizer, encoder, and task models, and instantiates and returns a JiantModel.

    Args:
        model_type (str): model shortcut name.
        model_config_path (str): Path to the JSON file containing the configuration parameters.
        tokenizer_path (str): path to tokenizer directory.
        task_dict (Dict[str, tasks.Task]): map from task name to task instance.
        taskmodels_config: maps mapping from tasks to models, and specifying task-model configs.

    Returns:
        JiantModel nn.Module.

    """
    model_arch = ModelArchitectures.from_model_type(model_type)
    transformers_class_spec = TRANSFORMERS_CLASS_SPEC_DICT[model_arch]
    tokenizer = model_setup.get_tokenizer(model_type=model_type, tokenizer_path=tokenizer_path)
    ancestor_model = get_ancestor_model(
        transformers_class_spec=transformers_class_spec, model_config_path=model_config_path,
    )
    encoder = get_encoder(model_arch=model_arch, ancestor_model=ancestor_model)
    taskmodels_dict = {
        taskmodel_name: create_taskmodel(
            task=task_dict[task_name_list[0]],  # Take the first task
            model_arch=model_arch,
            encoder=encoder,
            taskmodel_kwargs=taskmodels_config.get_taskmodel_kwargs(taskmodel_name),
        )
        for taskmodel_name, task_name_list in get_taskmodel_and_task_names(
            taskmodels_config.task_to_taskmodel_map
        ).items()
    }
    return primary.JiantModel(
        task_dict=task_dict,
        encoder=encoder,
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
    model_arch = ModelArchitectures.from_encoder(encoder=encoder)
    encoder_prefix = MODEL_PREFIX[model_arch] + "."
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
    model_arch = get_model_arch_from_jiant_model(jiant_model=jiant_model)
    if model_arch == ModelArchitectures.BERT:
        mlm_weights_map = {
            "bias": "cls.predictions.bias",
            "dense.weight": "cls.predictions.transform.dense.weight",
            "dense.bias": "cls.predictions.transform.dense.bias",
            "LayerNorm.weight": "cls.predictions.transform.LayerNorm.weight",
            "LayerNorm.bias": "cls.predictions.transform.LayerNorm.bias",
            "decoder.weight": "cls.predictions.decoder.weight",
            "decoder.bias": "cls.predictions.bias",  # <-- linked directly to bias
        }
        mlm_weights_dict = {new_k: weights_dict[old_k] for new_k, old_k in mlm_weights_map.items()}
    elif model_arch in (ModelArchitectures.ROBERTA, ModelArchitectures.XLM_ROBERTA):
        mlm_weights_dict = {
            strings.remove_prefix(k, "lm_head."): v for k, v in weights_dict.items()
        }
        mlm_weights_dict["decoder.bias"] = mlm_weights_dict["bias"]
    elif model_arch == ModelArchitectures.ALBERT:
        mlm_weights_dict = {
            strings.remove_prefix(k, "predictions."): v for k, v in weights_dict.items()
        }
    else:
        raise KeyError(model_arch)
    missed = set()
    for taskmodel_name, taskmodel in jiant_model.taskmodels_dict.items():
        if not isinstance(taskmodel, taskmodels.MLMModel):
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


def create_taskmodel(
    task, model_arch, encoder, taskmodel_kwargs: Optional[Dict] = None
) -> taskmodels.Taskmodel:
    """Creates, initializes and returns the task model for a given task type and encoder.

    Args:
        task (Task): Task object associated with the taskmodel being created.
        model_arch (ModelArchitectures.Any): Model architecture (e.g., ModelArchitectures.BERT).
        encoder (PreTrainedModel): Transformer w/o heads (embedding layer + self-attention layer).
        taskmodel_kwargs (Optional[Dict]): map containing any kwargs needed for taskmodel setup.

    Raises:
        KeyError if task does not have valid TASK_TYPE.

    Returns:
        Taskmodel (e.g., ClassificationModel) appropriate for the task type and encoder.

    """
    if model_arch in [
        ModelArchitectures.BERT,
        ModelArchitectures.ROBERTA,
        ModelArchitectures.ALBERT,
        ModelArchitectures.XLM_ROBERTA,
        ModelArchitectures.ELECTRA,
    ]:
        hidden_size = encoder.config.hidden_size
        hidden_dropout_prob = encoder.config.hidden_dropout_prob
    elif model_arch in [
        ModelArchitectures.BART,
        ModelArchitectures.MBART,
    ]:
        hidden_size = encoder.config.d_model
        hidden_dropout_prob = encoder.config.dropout
    else:
        raise KeyError()

    if task.TASK_TYPE == TaskTypes.CLASSIFICATION:
        assert taskmodel_kwargs is None
        classification_head = heads.ClassificationHead(
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            num_labels=len(task.LABELS),
        )
        taskmodel = taskmodels.ClassificationModel(
            encoder=encoder, classification_head=classification_head,
        )
    elif task.TASK_TYPE == TaskTypes.REGRESSION:
        assert taskmodel_kwargs is None
        regression_head = heads.RegressionHead(
            hidden_size=hidden_size, hidden_dropout_prob=hidden_dropout_prob,
        )
        taskmodel = taskmodels.RegressionModel(encoder=encoder, regression_head=regression_head)
    elif task.TASK_TYPE == TaskTypes.MULTIPLE_CHOICE:
        assert taskmodel_kwargs is None
        choice_scoring_head = heads.RegressionHead(
            hidden_size=hidden_size, hidden_dropout_prob=hidden_dropout_prob,
        )
        taskmodel = taskmodels.MultipleChoiceModel(
            encoder=encoder, num_choices=task.NUM_CHOICES, choice_scoring_head=choice_scoring_head,
        )
    elif task.TASK_TYPE == TaskTypes.SPAN_PREDICTION:
        assert taskmodel_kwargs is None
        span_prediction_head = heads.TokenClassificationHead(
            hidden_size=hidden_size,
            hidden_dropout_prob=encoder.config.hidden_dropout_prob,
            num_labels=2,
        )
        taskmodel = taskmodels.SpanPredictionModel(
            encoder=encoder, span_prediction_head=span_prediction_head,
        )
    elif task.TASK_TYPE == TaskTypes.SPAN_COMPARISON_CLASSIFICATION:
        assert taskmodel_kwargs is None
        span_comparison_head = heads.SpanComparisonHead(
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            num_spans=task.num_spans,
            num_labels=len(task.LABELS),
        )
        taskmodel = taskmodels.SpanComparisonModel(
            encoder=encoder, span_comparison_head=span_comparison_head,
        )
    elif task.TASK_TYPE == TaskTypes.MULTI_LABEL_SPAN_CLASSIFICATION:
        assert taskmodel_kwargs is None
        span_comparison_head = heads.SpanComparisonHead(
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            num_spans=task.num_spans,
            num_labels=len(task.LABELS),
        )
        taskmodel = taskmodels.MultiLabelSpanComparisonModel(
            encoder=encoder, span_comparison_head=span_comparison_head,
        )
    elif task.TASK_TYPE == TaskTypes.TAGGING:
        assert taskmodel_kwargs is None
        token_classification_head = heads.TokenClassificationHead(
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            num_labels=len(task.LABELS),
        )
        taskmodel = taskmodels.TokenClassificationModel(
            encoder=encoder, token_classification_head=token_classification_head,
        )
    elif task.TASK_TYPE == TaskTypes.SQUAD_STYLE_QA:
        assert taskmodel_kwargs is None
        qa_head = heads.QAHead(hidden_size=hidden_size)
        taskmodel = taskmodels.QAModel(encoder=encoder, qa_head=qa_head)
    elif task.TASK_TYPE == TaskTypes.MASKED_LANGUAGE_MODELING:
        assert taskmodel_kwargs is None
        if model_arch == ModelArchitectures.BERT:
            mlm_head = heads.BertMLMHead(
                hidden_size=hidden_size,
                vocab_size=encoder.config.vocab_size,
                layer_norm_eps=encoder.config.layer_norm_eps,
                hidden_act=encoder.config.hidden_act,
            )
        elif model_arch == ModelArchitectures.ROBERTA:
            mlm_head = heads.RobertaMLMHead(
                hidden_size=hidden_size,
                vocab_size=encoder.config.vocab_size,
                layer_norm_eps=encoder.config.layer_norm_eps,
            )
        elif model_arch == ModelArchitectures.ALBERT:
            mlm_head = heads.AlbertMLMHead(
                hidden_size=hidden_size,
                embedding_size=encoder.config.embedding_size,
                vocab_size=encoder.config.vocab_size,
                hidden_act=encoder.config.hidden_act,
            )
        elif model_arch == ModelArchitectures.XLM_ROBERTA:
            mlm_head = heads.RobertaMLMHead(
                hidden_size=hidden_size,
                vocab_size=encoder.config.vocab_size,
                layer_norm_eps=encoder.config.layer_norm_eps,
            )
        elif model_arch in (
            ModelArchitectures.BART,
            ModelArchitectures.MBART,
            ModelArchitectures.ELECTRA,
        ):
            raise NotImplementedError()
        else:
            raise KeyError(model_arch)
        taskmodel = taskmodels.MLMModel(encoder=encoder, mlm_head=mlm_head)
    elif task.TASK_TYPE == TaskTypes.EMBEDDING:
        if taskmodel_kwargs["pooler_type"] == "mean":
            pooler_head = heads.MeanPoolerHead()
        elif taskmodel_kwargs["pooler_type"] == "first":
            pooler_head = heads.FirstPoolerHead()
        else:
            raise KeyError(taskmodel_kwargs["pooler_type"])
        taskmodel = taskmodels.EmbeddingModel(
            encoder=encoder, pooler_head=pooler_head, layer=taskmodel_kwargs["layer"],
        )
    else:
        raise KeyError(task.TASK_TYPE)
    return taskmodel


def get_encoder(model_arch, ancestor_model):
    """From model architecture, get the encoder (encoder = embedding layer + self-attention layer).

    This function will return the "The bare Bert Model transformer outputting raw hidden-states
    without any specific head on top", when provided with ModelArchitectures and BertForPreTraining
    model. See Hugging Face's BertForPreTraining and BertModel documentation for more info.

    Args:
        model_arch: Model architecture.
        ancestor_model: Model with pretraining heads attached.

    Raises:
        KeyError if ModelArchitectures

    Returns:
        Bare pretrained model outputting raw hidden-states without a specific head on top.

    """
    if model_arch == ModelArchitectures.BERT:
        return ancestor_model.bert
    elif model_arch == ModelArchitectures.ROBERTA:
        return ancestor_model.roberta
    elif model_arch == ModelArchitectures.ALBERT:
        return ancestor_model.albert
    elif model_arch == ModelArchitectures.XLM_ROBERTA:
        return ancestor_model.roberta
    elif model_arch in (ModelArchitectures.BART, ModelArchitectures.MBART):
        return ancestor_model.model
    elif model_arch == ModelArchitectures.ELECTRA:
        return ancestor_model.electra
    else:
        raise KeyError(model_arch)


@dataclass
class TransformersClassSpec:
    config_class: Any
    tokenizer_class: Any
    model_class: Any


TRANSFORMERS_CLASS_SPEC_DICT = {
    ModelArchitectures.BERT: TransformersClassSpec(
        config_class=transformers.BertConfig,
        tokenizer_class=transformers.BertTokenizer,
        model_class=transformers.BertForPreTraining,
    ),
    ModelArchitectures.ROBERTA: TransformersClassSpec(
        config_class=transformers.RobertaConfig,
        tokenizer_class=transformers.RobertaTokenizer,
        model_class=transformers.RobertaForMaskedLM,
    ),
    ModelArchitectures.ALBERT: TransformersClassSpec(
        config_class=transformers.AlbertConfig,
        tokenizer_class=transformers.AlbertTokenizer,
        model_class=transformers.AlbertForMaskedLM,
    ),
    ModelArchitectures.XLM_ROBERTA: TransformersClassSpec(
        config_class=transformers.XLMRobertaConfig,
        tokenizer_class=transformers.XLMRobertaTokenizer,
        model_class=transformers.XLMRobertaForMaskedLM,
    ),
    ModelArchitectures.BART: TransformersClassSpec(
        config_class=transformers.BartConfig,
        tokenizer_class=transformers.BartTokenizer,
        model_class=transformers.BartForConditionalGeneration,
    ),
    ModelArchitectures.MBART: TransformersClassSpec(
        config_class=transformers.BartConfig,
        tokenizer_class=transformers.MBartTokenizer,
        model_class=transformers.BartForConditionalGeneration,
    ),
    ModelArchitectures.ELECTRA: TransformersClassSpec(
        config_class=transformers.ElectraConfig,
        tokenizer_class=transformers.ElectraTokenizer,
        model_class=transformers.ElectraForPreTraining,
    ),
}


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


def get_model_arch_from_jiant_model(jiant_model: nn.Module) -> ModelArchitectures:
    return ModelArchitectures.from_encoder(encoder=jiant_model.encoder)


MODEL_PREFIX = {
    ModelArchitectures.BERT: "bert",
    ModelArchitectures.ROBERTA: "roberta",
    ModelArchitectures.ALBERT: "albert",
    ModelArchitectures.XLM_ROBERTA: "xlm-roberta",
    ModelArchitectures.BART: "model",
    ModelArchitectures.MBART: "model",
    ModelArchitectures.ELECTRA: "electra",
}


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
