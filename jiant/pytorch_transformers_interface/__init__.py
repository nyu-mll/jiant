"""
Warning: jiant currently depends on *both* pytorch_pretrained_bert > 0.6 _and_
pytorch_transformers > 1.0

These are the same package, though the name changed between these two versions. AllenNLP requires
0.6 to support the BertAdam optimizer, and jiant directly requires 1.0 to support XLNet and
WWM-BERT.

This AllenNLP issue is relevant: https://github.com/allenai/allennlp/issues/3067

Note: huggingface forgot to upload bert-large-uncased-whole-word-masking-finetuned-squad
When they fix it, remove this note
https://github.com/huggingface/pytorch-transformers/issues/763 

TODO: We do not support non-English versions of XLM, if you need them, add some code in XLMEmbedderModule
to prepare langs input to pytorch_transformers.XLMModel
"""


def input_module_uses_pytorch_transformers(input_module):
    return (
        input_module.startswith("bert-")
        or input_module.startswith("roberta-")
        or input_module.startswith("xlnet-")
        or input_module.startswith("gpt2")
        or input_module.startswith("openai-gpt")
        or input_module.startswith("transfo-xl-")
        or input_module.startswith("xlm-")
    )


def input_module_tokenizer_name(input_module):
    input_module_to_pretokenized = {
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
        "bert-base-chinese": "bert_chinese",
        "bert-base-german-cased": "bert_german_cased",
        "roberta-base": "roberta",
        "roberta-large": "roberta",
        "roberta-large-mnli": "roberta",
        "xlnet-base-cased": "xlnet_cased",
        "xlnet-large-cased": "xlnet_cased",
        "openai-gpt": "openai_gpt",
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2",
        "gpt2-large": "gpt2",
        "transfo-xl-wt103": "transfo_xl",
        "xlm-mlm-en-2048": "xlm_en",
        "xlm-mlm-ende-1024": "xlm_ende",
        "xlm-mlm-enfr-1024": "xlm_enfr",
        "xlm-clm-enfr-1024": "xlm_enfr",
        "xlm-mlm-enro-1024": "xlm_enro",
        "xlm-mlm-tlm-xnli15-1024": "xlm_xnli",
        "xlm-mlm-xnli15-1024": "xlm_xnli",
    }
    return input_module_to_pretokenized[input_module]
