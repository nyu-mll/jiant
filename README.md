<div align="center">

# `jiant` is an NLP toolkit
**The multitask and transfer learning toolkit for natural language processing research**

[![Generic badge](https://img.shields.io/github/v/release/nyu-mll/jiant)](https://shields.io/)
[![codecov](https://codecov.io/gh/nyu-mll/jiant/branch/master/graph/badge.svg)](https://codecov.io/gh/nyu-mll/jiant)
[![CircleCI](https://circleci.com/gh/nyu-mll/jiant/tree/master.svg?style=shield)](https://circleci.com/gh/nyu-mll/jiant/tree/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

**Why should I use `jiant`?**
- `jiant` supports [multitask learning](https://colab.research.google.com/github/nyu-mll/jiant/examples/notebooks/jiant_Multi_Task_Example.ipynb)
- `jiant` supports [transfer learning](https://colab.research.google.com/github/nyu-mll/jiant/examples/notebooks/jiant_STILTs_Example.ipynb)
- `jiant` supports [50+ natural language understanding tasks](./md_docs/tasks.md)
- `jiant` supports the following benchmarks:
    - [GLUE](./guides/benchmarks/glue.md)
    - [SuperGLUE](./guides/benchmarks/superglue.md)
    - [XTREME](./guides/benchmarks/xtreme.md)
- `jiant` is a research library and users are encouraged to extend, change, and contribute to match their needs!

**A few additional things you might want to know about `jiant`:**
- `jiant` is configuration file driven
- `jiant` is built with [PyTorch](https://pytorch.org)
- `jiant` integrates with [`datasets`](https://github.com/huggingface/datasets) to manage task data
- `jiant` integrates with [`transformers`](https://github.com/huggingface/transformers) to manage models and tokenizers.


### Quick Introduction
The following example fine tunes a RoBERTa model on the MRPC dataset.

Python version:
```python
from jiant.proj.simple import runscript as run
import jiant.scripts.download_data.runscript as downloader

# Download the Data
downloader.download_data([“mrpc”], "/content/data")

# Set up the arguments for the Simple API
args = run.RunConfiguration(
   run_name="simple",
   exp_dir="/content/exp",
   data_dir="/content/data",
   model_type="roberta-base",
   tasks="mrpc",
   train_batch_size=16,
   num_train_epochs=3
)

# Run!
run.run_simple(args)
```

Bash version:
```bash
python jiant/scripts/download_data/runscript.py download --tasks mrpc --output_path /content/data
python jiant/proj/simple/runscript.py run --run_name=simple --exp_dir /content/data --data_dir /content/data --model_type roberta-base --tasks mrpc --train_batch_size 16 --num_train_epochs 3
```

Examples of more complex training workflows are found [here](./examples/README.md).

### Installation
To install `jiant` as a user:
```
pip install jiant
```
To install `jiant` as a developer:
```
git clone https://github.com/nyu-mll/jiant.git
cd jiant
pip install -e .
```
To check `jiant` was correctly installed, run a [simple example](./examples/notebooks/simple_api_fine_tuning.ipynb).


### Contributing
The `jiant` project's contributing guidelines can be found [here](CONTRIBUTING.md).

### Looking for `jiant v1.3.2`?
`jiant v1.3.2` has been moved to [jiant-v1-legacy](https://github.com/nyu-mll/jiant-v1-legacy) to support ongoing research with the library. `jiant v2.x.x` is more modular and scalable than `jiant v1.3.2` and has been designed to reflect the needs of the current NLP research community. We strongly recommended any new projects use `jiant v2.x.x`.

#### Papers using `jiant v1.3.2`

[`jiant`](https://github.com/nyu-mll/jiant-v1-legacy) has been used in these papers so far:

- [Can You Tell Me How to Get Past Sesame Street? Sentence-Level Pretraining Beyond Language Modeling](https://arxiv.org/abs/1812.10860) (formerly "Looking for ELMo's Friends")
- [What do you learn from context? Probing for sentence structure in contextualized word representations](https://openreview.net/forum?id=SJzSgnRcKX) ("edge probing")
- [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950) ("BERT layer paper")
- [Probing What Different NLP Tasks Teach Machines about Function Word Comprehension](https://arxiv.org/abs/1904.11544) ("function word probing")
- [Investigating BERT’s Knowledge of Language: Five Analysis Methods with NPIs](https://arxiv.org/abs/1909.02597) ("BERT NPI paper")

To exactly reproduce experiments from [the ELMo's Friends paper](https://arxiv.org/abs/1812.10860) use the [`jsalt-experiments`](https://github.com/jsalt18-sentence-repl/jiant/tree/jsalt-experiments) branch. That will contain a snapshot of the code as of early August, potentially with updated documentation.

For the [edge probing paper](https://openreview.net/forum?id=SJzSgnRcKX) and the [BERT layer paper](https://arxiv.org/abs/1905.05950), see the [probing/](https://github.com/nyu-mll/jiant-v1-legacy/tree/master/probing) directory.

For the [function word probing paper](https://arxiv.org/abs/1904.11544), use [this branch](https://github.com/nyu-mll/jiant/tree/naacl_probingpaper) and refer to the instructions in the [scripts/fwords/](https://github.com/nyu-mll/jiant/tree/naacl_probingpaper/scripts/fwords) directory.

For the [BERT NPI paper](https://arxiv.org/abs/1909.02597) follow the instructions in [scripts/bert_npi](https://github.com/nyu-mll/jiant/tree/blimp-and-npi/scripts/bert_npi) on the [`blimp-and-npi`](https://github.com/nyu-mll/jiant/tree/blimp-and-npi) branch.

### Citation

If you use `jiant ≥ v2.0.0` in academic work, please cite it directly:

```
@misc{phang2020jiant,
    author = {Jason Phang and Phil Yeres and Jesse Swanson and Haokun Liu and Ian F. Tenney and Phu Mon Htut and Clara Vania and Alex Wang and Samuel R. Bowman},
    title = {\texttt{jiant} 2.0: A software toolkit for research on general-purpose text understanding models},
    howpublished = {\url{http://jiant.info/}},
    year = {2020}
}
```

If you use `jiant ≤ v1.3.2` in academic work, please use the citation found [here](https://github.com/nyu-mll/jiant-v1-legacy).

### Acknowledgments

- This work was made possible in part by a donation to NYU from Eric and Wendy Schmidt made
by recommendation of the Schmidt Futures program, and by support from Intuit Inc.
- We gratefully acknowledge the support of NVIDIA Corporation with the donation of a Titan V GPU used at NYU in this work.
- Developer Jesse Swanson is supported by the Moore-Sloan Data Science Environment as part of the NYU Data Science Services initiative.

### License
`jiant` is released under the [MIT License](https://github.com/jiant-dev/jiant/blob/master/LICENSE).