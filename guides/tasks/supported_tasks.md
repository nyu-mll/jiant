# Tasks

## Supported Tasks

| Name | `task_name` | `jiant` | Downloader | `jiant_task_name` | Misc |
|---|---|:---:|:---:|---|---|
| MCTACO | mctaco | ✅ |  | mctaco |  |
| MCTest | mctest160 or mctest500 | ✅ |  | mctest |  |
| [Argument Reasoning Comprehension](https://arxiv.org/abs/1708.01425) | arct | ✅ |  | arct | [Github](https://github.com/UKPLab/argument-reasoning-comprehension-task) |
| Abductive NLI | abductive_nli | ✅ | ✅ | abductive_nli |  |
| SuperGLUE Winogender Diagnostic | superglue_axg | ✅ | ✅ | superglue_axg | SuperGLUE |
| Acceptability Definiteness | acceptability_definiteness | ✅ |  | acceptability_definiteness | Function Words |
| Adversarial NLI | `adversarial_nli_{round}` | ✅ |  | adversarial_nli | 3 rounds |
| ARC ("easy" version) | arc_easy | ✅ | ✅ | arc_easy | [site](https://allenai.org/data/arc) |
| ARC ("challenge" version) | arc_challenge | ✅ | ✅ | arc_challenge | [site](https://allenai.org/data/arc) |
| BoolQ | boolq | ✅ | ✅ | boolq | SuperGLUE |
| BUCC2018 | `bucc2018_{lang}` | ✅ | ✅ | bucc2018 | XTREME, multi-lang |
| CommitmentBank | cb | ✅ | ✅ | cb | SuperGLUE |
| CCG | ccg | ✅ |  | ccg |  |
| CoLA | cola | ✅ | ✅ | cola | GLUE |
| CommonsenseQA | commonsenseqa | ✅ | ✅ | commonsenseqa |  |
| EP-Const | nonterminal | ✅ |  | nonterminal | Edge-Probing |
| COPA | copa | ✅ | ✅ | copa | SuperGLUE |
| EP-Coref | coref | ✅ |  | coref | Edge-Probing |
| Cosmos QA | cosmosqa | ✅ | ✅ | cosmosqa |  |
| EP-UD | dep | ✅ |  | dep | Edge-Probing |
| EP-DPR | dpr | ✅ |  | dpr | Edge-Probing |
| Fever NLI | fever_nli | ✅ | ✅ | fever_nli |  |
| GLUE Diagnostic | glue_diagnostics | ✅ | ✅ | glue_diagnostics | GLUE |
| HellaSwag | hellaswag | ✅ | ✅ | hellaswag |  |
| [MCScript2.0](https://arxiv.org/pdf/1905.09531.pdf) | mcscript | ✅ |  | mcscript | [data](https://my.hidrive.com/share/wdnind8pp5#$/) |
| MLM | * | ✅ | * | mlm_simple | See task-specific notes. |
| MLQA | `mlqa_{lang1}_{lang2}` | ✅ | ✅ | mlqa | XTREME, multi-lang |
| MNLI | mnli | ✅ | ✅ | mnli | GLUE, MNLI-matched |
| MNLI-mismatched | mnli_mismatched | ✅ | ✅ | mnli_mismatched | GLUE |
| MRPC | mrpc | ✅ | ✅ | mrpc | GLUE |
| MultiRC | multirc | ✅ | ✅ | multirc | SuperGLUE |
| Mutual (standard version) | mutual | ✅ |  | mutual | [site](https://github.com/Nealcly/MuTual) |
| Mutual ("challenge" version) | mutual_plus | ✅ |  | mutual_plus | [site](https://github.com/Nealcly/MuTual) |
| Natural Questions | mrqa_natural_questions | ✅ | ✅ | mrqa_natural_questions | [MRQA](https://mrqa.github.io/) version of task |
| NewsQA | newsqa | ✅ | ✅ | newsqa |  |
| PIQA | piqa | ✅ | ✅ | piqa | [PIQA](https://yonatanbisk.com/piqa/) |
| QAMR | qamr | ✅ | ✅ | qamr |  |
| QA-SRL | qasrl | ✅ | ✅ | qasrl |  |
| Quail | quail | ✅ | | quail | [site](http://text-machine.cs.uml.edu/lab2/projects/quail/) |
| Quoref | quoref | ✅ | ✅ | quoref |  |
| EP-NER | ner | ✅ |  | ner | Edge-Probing |
| PAWS-X | `pawsx_{lang}` | ✅ | ✅ | pawsx | XTREME, multi-lang |
| WikiAnn | `panx_{lang}` | ✅ | ✅ | panx | XTREME, multi-lang |
| EP-POS | pos | ✅ |  | pos | Edge-Probing |
| QNLI | qnli | ✅ | ✅ | qnli | GLUE |
| QQP | qqp | ✅ | ✅ | qqp | GLUE |
| ReCord | record | ✅ | ✅ | record | SuperGLUE |
| RTE | rte | ✅ | ✅ | rte | GLUE, SuperGLUE |
| SciTail | scitail | ✅ | ✅ | scitail |  |
| SentEval: Tense | senteval_tense | ✅ |  | senteval_tense | SentEval |
| EP-Rel | semeval | ✅ |  | semeval | Edge-Probing |
| SNLI | snli | ✅ | ✅ | snli |  |
| SocialIQA | socialiqa | ✅ | ✅ | socialiqa |  |
| EP-SPR1 | spr1 | ✅ |  | spr1 | Edge-Probing |
| EP-SPR2 | spr2 | ✅ |  | spr2 | Edge-Probing |
| SQuAD 1.1 | squad_v1 | ✅ | ✅ | squad |  |
| SQuAD 2.0 | squad_v2 | ✅ | ✅ | squad |  |
| EP-SRL | srl | ✅ |  | srl | Edge-Probing |
| SST-2 | sst | ✅ | ✅ | sst | GLUE |
| STS-B | stsb | ✅ | ✅ | stsb | GLUE |
| SuperGLUE Broad Coverage Diagnostic | superglue_axg | ✅ | ✅ | superglue_axg | SuperGLUE |
| SWAG | swag | ✅ | ✅ | swag |  |
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
