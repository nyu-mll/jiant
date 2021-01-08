import os

import torch
from transformers import AutoModelForPreTraining, AutoTokenizer, AutoConfig

import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    hf_pretrained_model_name = zconf.attr(type=str)
    output_base_path = zconf.attr(type=str)


def export_model(
    hf_pretrained_model_name: str, output_base_path: str,
):
    """Retrieve model and tokenizer from Transformers and save all necessary data
    Things saved:
    - Model weights
    - Model config JSON (corresponding to corresponding Transformers model Config object)
    - Tokenizer data
    - JSON file pointing to paths for the above
    Args:
        hf_pretrained_model_name (:obj:`str`): A string, the `model id` of a pretrained model hosted
                                            inside a model repo on huggingface.co. Valid model
                                            ids can be located at the root-level, like
                                            ``bert-base-uncased``, or namespaced under a user or
                                            organization name, like ``dbmdz/bert-base-german-cased``
        output_base_path: Base path to save output to
        model_class: Model class
        tokenizer_class: Tokenizer class
        hf_model_name: (Optional) hf_model_name from https://huggingface.co/models,
                       if it differs from model_type
    """
    model_fol_path = os.path.join(output_base_path, "model")

    config = AutoConfig.from_pretrained(hf_pretrained_model_name)
    model_type = config.model_type
    model_path = os.path.join(model_fol_path, f"{model_type}.p")
    model_config_path = os.path.join(model_fol_path, f"{model_type}.json")
    tokenizer_fol_path = os.path.join(output_base_path, "tokenizer")

    os.makedirs(tokenizer_fol_path, exist_ok=True)
    os.makedirs(model_fol_path, exist_ok=True)

    model = AutoModelForPreTraining.from_pretrained(hf_pretrained_model_name)
    torch.save(model.state_dict(), model_path)
    py_io.write_json(model.config.to_dict(), model_config_path)
    tokenizer = AutoTokenizer.from_pretrained(hf_pretrained_model_name)
    tokenizer.save_pretrained(tokenizer_fol_path)
    config = {
        "model_type": model_type,
        "model_path": model_path,
        "model_config_path": model_config_path,
        "model_tokenizer_path": tokenizer_fol_path,
    }
    py_io.write_json(config, os.path.join(output_base_path, f"config.json"))


def main():
    args = RunConfiguration.default_run_cli()
    export_model(
        hf_pretrained_model_name=args.hf_pretrained_model_name, output_base_path=args.output_base_path,
    )


if __name__ == "__main__":
    main()
