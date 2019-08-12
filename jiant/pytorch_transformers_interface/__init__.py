"""
Warning: jiant currently depends on *both* pytorch_pretrained_bert > 0.6 _and_
pytorch_transformers > 1.0

These are the same package, though the name changed between these two versions. AllenNLP requires
0.6 to support the BertAdam optimizer, and jiant directly requires 1.0 to support XLNet and
WWM-BERT.

This AllenNLP issue is relevant: https://github.com/allenai/allennlp/issues/3067
"""
from jiant.pytorch_transformers_interface import modules

def input_module_support_pair_embedding(module_name):
    return module_name.startswith("bert-") or module_name.startswith("xlnet-")

def input_module_uses_pytorch_transformers(module_name):
    return module_name.startswith("bert-") or module_name.startswith("xlnet-") or \
           module_name.startswith("gpt2") or module_name.startswith("openai-gpt") or \
           module_name.startswith("transfo-xl-") or module_name.startswith("xlm-")

def input_module_tokenized_name(module_name):
    return "pytorch_transformers_%s_pretokenized" % module_name
