"""
Warning: jiant currently depends on *both* pytorch_pretrained_bert > 0.6 _and_
transformers > 2.3

These are the same package, though the name changed between these two versions. AllenNLP requires
0.6 to support the BertAdam optimizer, and jiant directly requires 2.3.

This AllenNLP issue is relevant: https://github.com/allenai/allennlp/issues/3067

TODO: We do not support non-English versions of XLM, if you need them, add some code in XLMEmbedderModule
to prepare langs input to transformers.XLMModel
"""

# All the supported input_module from huggingface transformers
# input_modules mapped to the same string share vocabulary
transformer_input_module_to_tokenizer_name = {
    "bert-base-uncased": "bert_uncased",
    "bert-large-uncased": "bert_uncased",
    "bert-large-uncased-whole-word-masking": "bert_uncased",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "bert_uncased",
    "bert-base-cased": "bert_cased",
    "bert-large-cased": "bert_cased",
    "bert-large-cased-whole-word-masking": "bert_cased",
    "bert-large-cased-whole-word-masking-finetuned-squad": "bert_cased",
    "bert-base-cased-finetuned-mrpc": "bert_cased",
    "bert-base-multilingual-uncased": "bert_multilingual_uncased",
    "bert-base-multilingual-cased": "bert_multilingual_cased",
    "roberta-base": "roberta",
    "roberta-large": "roberta",
    "roberta-large-mnli": "roberta",
    "xlnet-base-cased": "xlnet_cased",
    "xlnet-large-cased": "xlnet_cased",
    "openai-gpt": "openai_gpt",
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2",
    "gpt2-large": "gpt2",
    "gpt2-xl": "gpt2",
    "transfo-xl-wt103": "transfo_xl",
    "xlm-mlm-en-2048": "xlm_en",
    "xlm-roberta-base": "xlm_roberta",
    "xlm-roberta-large": "xlm_roberta",
    "albert-base-v1": "albert",
    "albert-large-v1": "albert",
    "albert-xlarge-v1": "albert",
    "albert-xxlarge-v1": "albert",
    "albert-base-v2": "albert",
    "albert-large-v2": "albert",
    "albert-xlarge-v2": "albert",
    "albert-xxlarge-v2": "albert",
}


def input_module_uses_transformers(input_module):
    return input_module in transformer_input_module_to_tokenizer_name


def input_module_tokenizer_name(input_module):
    return transformer_input_module_to_tokenizer_name[input_module]
