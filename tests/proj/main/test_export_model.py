import os
import pytest
from transformers import BertPreTrainedModel, BertTokenizer, RobertaForMaskedLM, RobertaTokenizer

import jiant.utils.python.io as py_io
from jiant.proj.main.export_model import export_model


@pytest.mark.parametrize(
    "model_type, model_class, tokenizer_class, hf_pretrained_model_name",
    [
        ("bert", BertPreTrainedModel, BertTokenizer, "bert-base-cased"),
        (
            "roberta",
            RobertaForMaskedLM,
            RobertaTokenizer,
            "nyu-mll/roberta-med-small-1M-1",
        ),
    ],
)
def test_export_model(tmp_path, model_type, model_class, tokenizer_class, hf_pretrained_model_name):
    export_model(
        hf_pretrained_model_name=hf_pretrained_model_name,
        output_base_path=tmp_path,
    )
    read_config = py_io.read_json(os.path.join(tmp_path, f"config.json"))
    assert read_config["model_type"] == model_type
    assert read_config["model_path"] == os.path.join(tmp_path, "model", f"{model_type}.p")
    assert read_config["model_config_path"] == os.path.join(tmp_path, "model", f"{model_type}.json")
    assert read_config["model_tokenizer_path"] == os.path.join(tmp_path, "tokenizer")
