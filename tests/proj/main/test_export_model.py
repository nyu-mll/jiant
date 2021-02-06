import os
import pytest
from transformers import BertPreTrainedModel, BertTokenizer, RobertaForMaskedLM, RobertaTokenizer

import jiant.utils.python.io as py_io
from jiant.proj.main.export_model import export_model


@pytest.mark.parametrize(
    "model_type, model_class, hf_pretrained_model_name_or_path",
    [
        ("bert", BertPreTrainedModel, "bert-base-cased"),
        ("roberta", RobertaForMaskedLM, "nyu-mll/roberta-med-small-1M-1",),
    ],
)
def test_export_model(tmp_path, model_type, model_class, hf_pretrained_model_name_or_path):
    export_model(
        hf_pretrained_model_name_or_path=hf_pretrained_model_name_or_path,
        output_base_path=tmp_path,
    )
    read_config = py_io.read_json(os.path.join(tmp_path, f"config.json"))
    assert read_config["hf_pretrained_model_name_or_path"] == hf_pretrained_model_name_or_path
    assert read_config["model_path"] == os.path.join(tmp_path, "model", "model.p")
    assert read_config["model_config_path"] == os.path.join(tmp_path, "model", "config.json")
