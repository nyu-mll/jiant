import os
import pytest
from transformers import BertPreTrainedModel, BertTokenizer, RobertaForMaskedLM, RobertaTokenizer

import jiant.utils.python.io as py_io
from jiant.proj.main.export_model import export_model


@pytest.mark.parametrize(
    "model_type, model_class, tokenizer_class, hf_model_name",
    [
        ("bert-base-cased", BertPreTrainedModel, BertTokenizer, "bert-base-cased"),
        (
            "roberta-med-small-1M-1",
            RobertaForMaskedLM,
            RobertaTokenizer,
            "nyu-mll/roberta-med-small-1M-1",
        ),
    ],
)
def test_export_model(tmp_path, model_type, model_class, tokenizer_class, hf_model_name):
    export_model(
        model_type=model_type,
        output_base_path=tmp_path,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        hf_model_name=hf_model_name,
    )
    read_config = py_io.read_json(os.path.join(tmp_path, f"config.json"))
    assert read_config["model_type"] == model_type
    assert read_config["model_path"] == os.path.join(tmp_path, "model", f"{model_type}.p")
    assert read_config["model_config_path"] == os.path.join(tmp_path, "model", f"{model_type}.json")
    assert read_config["model_tokenizer_path"] == os.path.join(tmp_path, "tokenizer")
