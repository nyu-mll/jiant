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
- `jiant` supports [multitask learning](https://colab.research.google.com/github/nyu-mll/jiant/blob/master/examples/notebooks/jiant_Multi_Task_Example.ipynb)
- `jiant` supports [transfer learning](https://colab.research.google.com/github/nyu-mll/jiant/blob/master/examples/notebooks/jiant_STILTs_Example.ipynb)
- `jiant` supports [50+ natural language understanding tasks](./guides/tasks/supported_tasks.md)
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

## Getting Started

* Get started with some simple [Examples](./examples)
* Learn more about `jiant` by reading our [Guides](./guides)
* See our [list of supported tasks](./guides/tasks/supported_tasks.md)

## Installation

To import `jiant` from source (recommended for researchers):
```bash
git clone https://github.com/nyu-mll/jiant.git
cd jiant
pip install -r requirements.txt

# Add the following to your .bash_rc or .bash_profile 
export PYTHONPATH=/path/to/jiant:$PYTHONPATH
```
If you plan to contribute to jiant, install additional dependencies with `pip install -r requirements-dev.txt`.

To install `jiant` from source (alternative for researchers):
```
git clone https://github.com/nyu-mll/jiant.git
cd jiant
pip install . -e
```

To install `jiant` from pip (recommended if you just want to train/use a model):
```
pip install jiant
```

We recommended that you install `jiant` in a virtual environment or a conda environment.

To check `jiant` was correctly installed, run a [simple example](./examples/notebooks/simple_api_fine_tuning.ipynb).


## Quick Introduction
The following example fine-tunes a RoBERTa model on the MRPC dataset.

Python version:
```python
from jiant.proj.simple import runscript as run
import jiant.scripts.download_data.runscript as downloader

# Download the Data
downloader.download_data(["mrpc"], "/content/data")

# Set up the arguments for the Simple API
args = run.RunConfiguration(
   run_name="simple",
   exp_dir="/path/to/exp",
   data_dir="/path/to/exp/tasks",
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
python jiant/scripts/download_data/runscript.py \
    download \
    --tasks mrpc \
    --output_path /path/to/exp/tasks
python jiant/proj/simple/runscript.py \
    run \
    --run_name simple \
    --exp_dir /path/to/exp \
    --data_dir /path/to/exp/tasks \
    --model_type roberta-base \
    --tasks mrpc \
    --train_batch_size 16 \
    --num_train_epochs 3
```

Examples of more complex training workflows are found [here](./examples/).


## Contributing
The `jiant` project's contributing guidelines can be found [here](CONTRIBUTING.md).

## Looking for `jiant v1.3.2`?
`jiant v1.3.2` has been moved to [jiant-v1-legacy](https://github.com/nyu-mll/jiant-v1-legacy) to support ongoing research with the library. `jiant v2.x.x` is more modular and scalable than `jiant v1.3.2` and has been designed to reflect the needs of the current NLP research community. We strongly recommended any new projects use `jiant v2.x.x`.

`jiant 1.x` has been used in in several papers. For instructions on how to reproduce papers by `jiant` authors that refer readers to this site for documentation (including Tenney et al., Wang et al., Bowman et al., Kim et al., Warstadt et al.), refer to the [jiant-v1-legacy](https://github.com/nyu-mll/jiant-v1-legacy) README.

## Citation

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

## Acknowledgments

- This work was made possible in part by a donation to NYU from Eric and Wendy Schmidt made
by recommendation of the Schmidt Futures program, and by support from Intuit Inc.
- We gratefully acknowledge the support of NVIDIA Corporation with the donation of a Titan V GPU used at NYU in this work.
- Developer Jesse Swanson is supported by the Moore-Sloan Data Science Environment as part of the NYU Data Science Services initiative.

## License
`jiant` is released under the [MIT License](https://github.com/nyu-mll/jiant/blob/master/LICENSE).
