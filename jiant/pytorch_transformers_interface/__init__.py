"""
Warning: jiant currently depends on *both* pytorch_pretrained_bert > 0.6 _and_
pytorch_transformers > 1.0

These are the same package, though the name changed between these two versions. AllenNLP requires
0.6 to support the BertAdam optimizer, and jiant directly requires 1.0 to support XLNet and
WWM-BERT.

This AllenNLP issue is relevant: https://github.com/allenai/allennlp/issues/3067
"""


def input_module_uses_pytorch_transformers(module_name):
    return module_name.startswith("bert-") or module_name.startswith("xlnet-")
