import os

import torch
import transformers

import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    model_type = zconf.attr(type=str)
    output_base_path = zconf.attr(type=str)


def lookup_and_export_model(model_type, output_base_path):
    model_class, tokenizer_class = get_model_and_tokenizer_classes(model_type)
    export_model(
        model_type=model_type,
        output_base_path=output_base_path,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
    )


def export_model(model_type, output_base_path, model_class, tokenizer_class):
    tokenizer_fol_path = os.path.join(output_base_path, "tokenizer")
    model_fol_path = os.path.join(output_base_path, "model")
    os.makedirs(tokenizer_fol_path, exist_ok=True)
    os.makedirs(model_fol_path, exist_ok=True)

    model_path = os.path.join(model_fol_path, f"{model_type}.p")
    model_config_path = os.path.join(model_fol_path, f"{model_type}.json")
    model = model_class.from_pretrained(model_type)
    torch.save(model.state_dict(), model_path)
    py_io.write_json(model.config.to_dict(), model_config_path)
    tokenizer = tokenizer_class.from_pretrained(model_type)
    tokenizer.save_pretrained(tokenizer_fol_path)
    config = {
        "model_type": model_type,
        "model_path": model_path,
        "model_config_path": model_config_path,
        "model_tokenizer_path": tokenizer_fol_path,
    }
    py_io.write_json(config, os.path.join(output_base_path, f"config.json"))


def get_model_and_tokenizer_classes(model_type):
    class_lookup = {
        "bert": (transformers.BertForPreTraining, transformers.BertTokenizer),
        "xlm-clm-": (transformers.XLMWithLMHeadModel, transformers.XLMTokenizer),
        "roberta": (transformers.RobertaForMaskedLM, transformers.RobertaTokenizer),
        "albert": (transformers.AlbertForMaskedLM, transformers.AlbertTokenizer),
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
        model_type=args.model_type, output_base_path=args.output_base_path,
    )


if __name__ == "__main__":
    main()
