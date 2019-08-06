"""
Warning: jiant currently depends on *both* pytorch_pretrained_bert > 0.6 _and_
pytorch_transformers > 1.0

These are the same package, though the name changed between these two versions. AllenNLP requires
0.6 to support the BertAdam optimizer, and jiant directly requires 1.0 to support XLNet and
WWM-BERT.

This AllenNLP issue is relevant: https://github.com/allenai/allennlp/issues/3067
"""

def input_module_support_bi_sentence(module_name):
    return module_name.startswith("bert") or module_name.startswith("xlnet")

def input_module_uses_pytorch_transformers(module_name):
    return module_name.startswith("bert") or module_name.startswith("xlnet") or module_name.startswith("gpt2") or module_name.startswith("openai-gpt")

def input_module_tokenized_name(module_name):
    if module_name.startswith("bert") or module_name.startswith("xlnet"):
        return "pytorch_transformers_wpm_pretokenized"
    elif module_name.startswith("gpt2"):
        return "pytorch_transformers_bytebpe_pretokenized"
    elif module_name.startswith("openai-gpt"):
        return "pytorch_transformers_bpe_pretokenized"