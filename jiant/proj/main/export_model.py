import os
from typing import Tuple, Type

import torch
import transformers

import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    model_type = zconf.attr(type=str)
    output_base_path = zconf.attr(type=str)
    hf_model_name = zconf.attr(type=str, default=None)


def lookup_and_export_model(model_type: str, output_base_path: str, hf_model_name: str = None):
    model_class, tokenizer_class = get_model_and_tokenizer_classes(model_type)
    export_model(
        model_type=model_type,
        output_base_path=output_base_path,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        hf_model_name=hf_model_name,
    )


def export_model(
    model_type: str,
    output_base_path: str,
    model_class: Type[transformers.PreTrainedModel],
    tokenizer_class: Type[transformers.PreTrainedTokenizer],
    hf_model_name: str = None,
):
    """Retrieve model and tokenizer from Transformers and save all necessary data
    Things saved:
    - Model weights
    - Model config JSON (corresponding to corresponding Transformers model Config object)
    - Tokenizer data
    - JSON file pointing to paths for the above
    Args:
        model_type: Model-type string. See: `get_model_and_tokenizer_classes`
        output_base_path: Base path to save output to
        model_class: Model class
        tokenizer_class: Tokenizer class
        hf_model_name: (Optional) hf_model_name from https://huggingface.co/models,
                       if it differs from model_type
    """
    if hf_model_name is None:
        hf_model_name = model_type

    tokenizer_fol_path = os.path.join(output_base_path, "tokenizer")
    model_fol_path = os.path.join(output_base_path, "model")
    os.makedirs(tokenizer_fol_path, exist_ok=True)
    os.makedirs(model_fol_path, exist_ok=True)

    model_path = os.path.join(model_fol_path, f"{model_type}.p")
    model_config_path = os.path.join(model_fol_path, f"{model_type}.json")
    model = model_class.from_pretrained(hf_model_name)
    torch.save(model.state_dict(), model_path)
    py_io.write_json(model.config.to_dict(), model_config_path)
    tokenizer = tokenizer_class.from_pretrained(hf_model_name)
    tokenizer.save_pretrained(tokenizer_fol_path)
    config = {
        "model_type": model_type,
        "model_path": model_path,
        "model_config_path": model_config_path,
        "model_tokenizer_path": tokenizer_fol_path,
    }
    py_io.write_json(config, os.path.join(output_base_path, f"config.json"))


def get_model_and_tokenizer_classes(
    model_type: str,
) -> Tuple[Type[transformers.PreTrainedModel], Type[transformers.PreTrainedTokenizer]]:
    # We want the chosen model to have all the weights from pretraining (if possible)
    class_lookup = {
        "bert": (transformers.BertForPreTraining, transformers.BertTokenizer),
        "xlm-clm-": (transformers.XLMWithLMHeadModel, transformers.XLMTokenizer),
        "roberta": (transformers.RobertaForMaskedLM, transformers.RobertaTokenizer),
        "albert": (transformers.AlbertForMaskedLM, transformers.AlbertTokenizer),
        "bart": (transformers.BartForConditionalGeneration, transformers.BartTokenizer),
        "mbart": (transformers.BartForConditionalGeneration, transformers.MBartTokenizer),
        "electra": (transformers.ElectraForPreTraining, transformers.ElectraTokenizer),
    }
    if model_type.split("-")[0] in class_lookup:
        return class_lookup[model_type.split("-")[0]]
    elif model_type.startswith("xlm-mlm-") or model_type.startswith("xlm-clm-"):
        return transformers.XLMWithLMHeadModel, transformers.XLMTokenizer
    elif model_type.startswith("xlm-roberta-"):
        return transformers.XLMRobertaForMaskedLM, transformers.XLMRobertaTokenizer
    else:
        raise KeyError()


def main():
    args = RunConfiguration.default_run_cli()
    lookup_and_export_model(
        model_type=args.model_type,
        output_base_path=args.output_base_path,
        hf_model_name=args.hf_model_name,
    )


if __name__ == "__main__":
    main()
