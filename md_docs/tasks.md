# Tasks

## Supported Tasks

| Name | `task_name` | `jiant` | Downloader | `jiant_task_name` | Misc | 
|---|---|:---:|:---:|---|---|
| Abductive NLI | abductive_nli | ✅ |  | abductive_nli |  |
| SuperGLUE Winogender Diagnostic | superglue_axg | ✅ | ✅ | superglue_axg | SuperGLUE |
| Acceptability Definiteness | acceptability_definiteness | ✅ |  | acceptability_definiteness | Function Words |
| Adversarial NLI | `adversarial_nli_{round}` | ✅ |  | adversarial_nli | 3 rounds |
| BoolQ | boolq | ✅ | ✅ | boolq | SuperGLUE |
| BUCC2018 | `bucc2018_{lang}` | ✅ | ✅ | bucc2018 | XTREME, multi-lang |
| CommitmentBank | cb | ✅ | ✅ | cb | SuperGLUE |
| CCG | ccg | ✅ |  | ccg |  |
| CoLA | cola | ✅ | ✅ | cola | GLUE |
| CommonsenseQA | commonsenseqa | ✅ | ✅ | commonsenseqa |  |
| EP-Const | nonterminal | ✅ |  | nonterminal | Edge-Probing |
| COPA | copa | ✅ | ✅ | copa | SuperGLUE |
| EP-Coref | coref | ✅ |  | coref | Edge-Probing |
| Cosmos QA | cosmosqa | ✅ |  | cosmosqa |  |
| EP-UD | dep | ✅ |  | dep | Edge-Probing |
| EP-DPR | dpr | ✅ |  | dpr | Edge-Probing |
| GLUE Diagnostic | glue_diagnostics | ✅ | ✅ | glue_diagnostics | GLUE |
| HellaSwag | hellaswag | ✅ | ✅ | hellaswag |  |
| MLM | * | ✅ |  | mlm_simple | See task-specific notes. |
| MLQA | `mlqa_{lang1}_{lang2}` | ✅ | ✅ | mlqa | XTREME, multi-lang |
| MNLI | mnli | ✅ | ✅ | mnli | GLUE, MNLI-matched |
| MNLI-mismatched | mnli_mismatched | ✅ | ✅ | mnli_mismatched | GLUE |
| MultiRC | multirc | ✅ | ✅ | multirc | SuperGLUE |
| MRPC | mrpc | ✅ | ✅ | mrpc | GLUE |
| QAMR | qamr | ✅ |  | qamr |  |
| QA-SRL | qa-srl | ✅ |  | qa-srl |  |
| EP-NER | ner | ✅ |  | ner | Edge-Probing |
| PAWS-X | `pawsx_{lang}` | ✅ | ✅ | pawsx | XTREME, multi-lang |
| WikiAnn | `panx_{lang}` | ✅ | ✅ | panx | XTREME, multi-lang |
| EP-POS | pos | ✅ |  | pos | Edge-Probing |
| QNLI | qnli | ✅ | ✅ | qnli | GLUE |
| QQP | qqp | ✅ | ✅ | qqp | GLUE |
| ReCord | record | ✅ | ✅ | record | SuperGLUE |
| RTE | rte | ✅ | ✅ | rte | GLUE, SuperGLUE |
| SciTail | scitail | ✅ |  | scitail |  |
| SentEval: Tense | senteval_tense | ✅ |  | senteval_tense | SentEval |
| EP-Rel | semeval | ✅ |  | semeval | Edge-Probing |
| SNLI | snli | ✅ | ✅ | snli |  |
| SocialIQA | socialiqa | ✅ |  | socialiqa |  |
| EP-SPR1 | spr1 | ✅ |  | spr1 | Edge-Probing |
| EP-SPR2 | spr2 | ✅ |  | spr2 | Edge-Probing |
| SQuAD 1.1 | squad_v1 | ✅ | ✅ | squad |  |
| SQuAD 2.0 | squad_v2 | ✅ | ✅ | squad |  |
| EP-SRL | srl | ✅ |  | srl | Edge-Probing |
| SST-2 | sst | ✅ | ✅ | sst | GLUE |
| STS-B | stsb | ✅ | ✅ | stsb | GLUE |
| SuperGLUE Broad Coverage Diagnostic | superglue_axg | ✅ | ✅ | superglue_axg | SuperGLUE |
| SWAG | swag | ✅ |  | swag |  |
| Tatoeba | `tatoeba_{lang}` | ✅ | ✅ | tatoeba | XTREME, multi-lang |
| TyDiQA | `tydiqa_{lang}` | ✅ | ✅ | tydiqa | XTREME, multi-lang |
| UDPOS | `udpos_{lang}` | ✅ | ✅ | udpos | XTREME, multi-lang |
| WiC | wic | ✅ | ✅ | wic | SuperGLUE |
| WNLI | wnli | ✅ | ✅ | wnli | GLUE |
| WSC | wsc | ✅ | ✅ | wsc | SuperGLUE |
| XNLI | `xnli_{lang}` | ✅ | ✅ | xnli | XTREME, multi-lang |
| XQuAD | `xquad_{lang}` | ✅ | ✅ | xquad | XTREME, multi-lang |

* `task_name`: Name-by-convention, used by downloader, and used in `JiantModel` to map from task names to task-models. You can change this as long as your settings are internally consistent. 
* `jiant`: Whether it's supported in `jiant` (i.e. you can train/eval on it)
* Downloader: Whether you can download using the downloader.
* `jiant_task_name`: Used to determine the programmatic behavior for the task (how to tokenize, what *kind* of task-model is compatible). Is tied directly to the code. See: `jiant.tasks.retrieval`. 

## Task-specific Notes

### Adversarial NLI

[Adversarial NLI](https://arxiv.org/pdf/1910.14599.pdf) has 3 rounds of adversarial data creation. A1/A2/A3 are expanding supersets of the previous round.


### Masked Language Modeling (MLM)

MLM is a generic task, implemented with the `jiant_task_name` "`mlm_simple`". In other words, it is meant to be used with any appropriately formatted file. 

`mlm_simple` expects input data files to be a single text file per phase, where each line corresponds to one example, and empty lines are ignored. This means that if a line corresponds to more than the `max_seq_length` of tokens during tokenization, everything past the first `max_seq_length` tokens per line will be ignored. We plan to add more complex implementations in the future.

You can structure your MLM task config file as follow:

```json
{
  "task": "mlm_simple",
  "paths": {
    "train": "/path/to/train.txt",
    "val": "/path/to/val.txt"
  },
  "name": "my_mlm_task"
}
```
