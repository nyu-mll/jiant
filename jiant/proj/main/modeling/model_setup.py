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
    weights_dict = torch.load(weights_path)
    return delegate_load(jiant_model=jiant_model, weights_dict=weights_dict, load_mode=load_mode)


def delegate_load(jiant_model, weights_dict: dict, load_mode: str):
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
    remainder_weights_dict = {}
    load_weights_dict = {}
    model_arch = get_model_arch_from_encoder(encoder=encoder)
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
        mlm_weights_dict = {
            "bias": "cls.predictions.bias",
            "dense.weight": "cls.predictions.transform.dense.weight",
            "dense.bias": "cls.predictions.transform.dense.bias",
            "LayerNorm.weight": "cls.predictions.transform.LayerNorm.weight",
            "LayerNorm.bias": "cls.predictions.transform.LayerNorm.bias",
            "decoder.weight": "cls.predictions.decoder.weight",
            # 'decoder.bias' <-- linked directly to bias
        }
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


def load_partial_heads(
    jiant_model, weights_dict, allow_missing_head_weights=False, allow_missing_head_model=False
):
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
    if task.TASK_TYPE == TaskTypes.CLASSIFICATION:
        assert taskmodel_kwargs is None
        classification_head = heads.ClassificationHead(
            hidden_size=encoder.config.hidden_size,
            hidden_dropout_prob=encoder.config.hidden_dropout_prob,
            num_labels=len(task.LABELS),
        )
        taskmodel = taskmodels.ClassificationModel(
            encoder=encoder, classification_head=classification_head,
        )
    elif task.TASK_TYPE == TaskTypes.REGRESSION:
        assert taskmodel_kwargs is None
        regression_head = heads.RegressionHead(
            hidden_size=encoder.config.hidden_size,
            hidden_dropout_prob=encoder.config.hidden_dropout_prob,
        )
        taskmodel = taskmodels.RegressionModel(encoder=encoder, regression_head=regression_head)
    elif task.TASK_TYPE == TaskTypes.MULTIPLE_CHOICE:
        assert taskmodel_kwargs is None
        choice_scoring_head = heads.RegressionHead(
            hidden_size=encoder.config.hidden_size,
            hidden_dropout_prob=encoder.config.hidden_dropout_prob,
        )
        taskmodel = taskmodels.MultipleChoiceModel(
            encoder=encoder, num_choices=task.NUM_CHOICES, choice_scoring_head=choice_scoring_head,
        )
    elif task.TASK_TYPE == TaskTypes.SPAN_COMPARISON_CLASSIFICATION:
        assert taskmodel_kwargs is None
        span_comparison_head = heads.SpanComparisonHead(
            hidden_size=encoder.config.hidden_size,
            hidden_dropout_prob=encoder.config.hidden_dropout_prob,
            num_spans=task.num_spans,
            num_labels=len(task.LABELS),
        )
        taskmodel = taskmodels.SpanComparisonModel(
            encoder=encoder, span_comparison_head=span_comparison_head,
        )
    elif task.TASK_TYPE == TaskTypes.TAGGING:
        assert taskmodel_kwargs is None
        token_classification_head = heads.TokenClassificationHead(
            hidden_size=encoder.config.hidden_size,
            hidden_dropout_prob=encoder.config.hidden_dropout_prob,
            num_labels=len(task.LABELS),
        )
        taskmodel = taskmodels.TokenClassificationModel(
            encoder=encoder, token_classification_head=token_classification_head,
        )
    elif task.TASK_TYPE == TaskTypes.SQUAD_STYLE_QA:
        assert taskmodel_kwargs is None
        qa_head = heads.QAHead(hidden_size=encoder.config.hidden_size)
        taskmodel = taskmodels.QAModel(encoder=encoder, qa_head=qa_head)
    elif task.TASK_TYPE == TaskTypes.MASKED_LANGUAGE_MODELING:
        assert taskmodel_kwargs is None
        if model_arch == ModelArchitectures.BERT:
            mlm_head = heads.BertMLMHead(
                hidden_size=encoder.config.hidden_size,
                vocab_size=encoder.config.vocab_size,
                layer_norm_eps=encoder.config.layer_norm_eps,
                hidden_act=encoder.config.hidden_act,
            )
        elif model_arch == ModelArchitectures.ROBERTA:
            mlm_head = heads.RobertaMLMHead(
                hidden_size=encoder.config.hidden_size,
                vocab_size=encoder.config.vocab_size,
                layer_norm_eps=encoder.config.layer_norm_eps,
            )
        elif model_arch == ModelArchitectures.ALBERT:
            mlm_head = heads.AlbertMLMHead(
                hidden_size=encoder.config.hidden_size,
                embedding_size=encoder.config.embedding_size,
                vocab_size=encoder.config.vocab_size,
                hidden_act=encoder.config.hidden_act,
            )
        elif model_arch == ModelArchitectures.XLM_ROBERTA:
            mlm_head = heads.RobertaMLMHead(
                hidden_size=encoder.config.hidden_size,
                vocab_size=encoder.config.vocab_size,
                layer_norm_eps=encoder.config.layer_norm_eps,
            )
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
}


def get_model_arch_from_encoder(encoder: nn.Module) -> ModelArchitectures:
    if type(encoder) is transformers.BertModel:
        return ModelArchitectures.BERT
    elif type(encoder) is transformers.RobertaModel:
        return ModelArchitectures.ROBERTA
    elif type(encoder) is transformers.AlbertModel:
        return ModelArchitectures.ALBERT
    elif type(encoder) is transformers.XLMRobertaModel:
        return ModelArchitectures.XLM_ROBERTA
    else:
        raise KeyError(type(encoder))


def get_taskmodel_and_task_names(task_to_taskmodel_map: Dict[str, str]) -> Dict[str, List[str]]:
    taskmodel_and_task_names = {}
    for task_name, taskmodel_name in task_to_taskmodel_map.items():
        if taskmodel_name not in taskmodel_and_task_names:
            taskmodel_and_task_names[taskmodel_name] = []
        taskmodel_and_task_names[taskmodel_name].append(task_name)
    return taskmodel_and_task_names


def get_model_arch_from_jiant_model(jiant_model: nn.Module) -> ModelArchitectures:
    return get_model_arch_from_encoder(encoder=jiant_model.encoder)


MODEL_PREFIX = {
    ModelArchitectures.BERT: "bert",
    ModelArchitectures.ROBERTA: "roberta",
    ModelArchitectures.ALBERT: "albert",
    ModelArchitectures.XLM_ROBERTA: "xlm-roberta",
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
