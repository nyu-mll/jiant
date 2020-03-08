# `jiant`

[![CircleCI](https://circleci.com/gh/nyu-mll/jiant/tree/master.svg?style=svg)](https://circleci.com/gh/nyu-mll/jiant/tree/master) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)


`jiant` is a software toolkit for natural language processing research, designed to facilitate work on multitask learning and transfer learning for sentence understanding tasks.

A few things you might want to know about `jiant`:

- `jiant` is configuration-driven. You can run an enormous variety of experiments by simply writing configuration files. Of course, if you need to add any major new features, you can also easily edit or extend the code.
- `jiant` contains implementations of strong baselines for the [GLUE](https://gluebenchmark.com) and [SuperGLUE](https://super.gluebenchmark.com/) benchmarks, and it's the recommended starting point for work on these benchmarks.
- `jiant` was developed at [the 2018 JSALT Workshop](https://www.clsp.jhu.edu/workshops/18-workshop/) by [the General-Purpose Sentence Representation Learning](https://jsalt18-sentence-repl.github.io/) team and is maintained by [the NYU Machine Learning for Language Lab](https://wp.nyu.edu/ml2/people/), with help from [many outside collaborators](https://github.com/nyu-mll/jiant/graphs/contributors) (especially Google AI Language's [Ian Tenney](https://ai.google/research/people/IanTenney)).
- `jiant` is built on [PyTorch](https://pytorch.org). It also uses many components from [AllenNLP](https://github.com/allenai/allennlp) and the HuggingFace Transformers [implementations](https://github.com/huggingface/transformers) for GPT, BERT and other transformer models.
- The name `jiant` doesn't mean much. The 'j' stands for JSALT. That's all the acronym we have.

For a full system overview of `jiant` version 1.3.0, see https://arxiv.org/abs/2003.02249.

## Getting Started

To find the setup instructions for using jiant and to run a simple example demo experiment using data from GLUE, follow this [getting started tutorial](https://github.com/nyu-mll/jiant/tree/master/tutorials/setup_tutorial.md)!

## Official Documentation

Our official documentation is here: https://jiant.info/documentation#/


## Running
To run an experiment, make a config file similar to `jiant/config/demo.conf` with your model configuration. In addition, you can use the `--overrides` flag to override specific variables. For example:
```sh
python main.py --config_file jiant/config/demo.conf \
    --overrides "exp_name = my_exp, run_name = foobar, d_hid = 256"
```
will run the demo config, but output to `$JIANT_PROJECT_PREFIX/my_exp/foobar`.
 To run the demo config, you will have to set environment variables. The best way to achieve that is to follow the instructions in [user_config_template.sh](user_config_template.sh)
*  `$JIANT_PROJECT_PREFIX`: the where the outputs will be saved.
*  `$JIANT_DATA_DIR`: location of the saved data. This is usually the location of the GLUE data in a simple default setup.
*  `$WORD_EMBS_FILE`: location of any word embeddings you want to use (not necessary when using ELMo, GPT, or BERT). You can download GloVe (840B) [here](http://nlp.stanford.edu/data/glove.840B.300d.zip) or fastText (2M) [here](https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip).
To have `user_config.sh` run automatically, follow instructions in [scripts/export_from_bash.sh](export_from_bash.sh).


## Suggested Citation

If you use `jiant` in academic work, please cite it directly:

```
@misc{wang2019jiant,
    author = {Alex Wang and Ian F. Tenney and Yada Pruksachatkun and Phil Yeres and Jason Phang and Haokun Liu and Phu Mon Htut and and Katherin Yu and Jan Hula and Patrick Xia and Raghu Pappagari and Shuning Jin and R. Thomas McCoy and Roma Patel and Yinghui Huang and Edouard Grave and Najoung Kim and Thibault F\'evry and Berlin Chen and Nikita Nangia and Anhad Mohananey and Katharina Kann and Shikha Bordia and Nicolas Patry and David Benton and Ellie Pavlick and Samuel R. Bowman},
    title = {\texttt{jiant} 1.3: A software toolkit for research on general-purpose text understanding models},
    howpublished = {\url{http://jiant.info/}},
    year = {2019}
}
```

## Papers

`jiant` has been used in these four papers so far:

- [Can You Tell Me How to Get Past Sesame Street? Sentence-Level Pretraining Beyond Language Modeling](https://arxiv.org/abs/1812.10860) (formerly "Looking for ELMo's Friends")
- [What do you learn from context? Probing for sentence structure in contextualized word representations](https://openreview.net/forum?id=SJzSgnRcKX) ("edge probing")
- [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950) ("BERT layer paper")
- [Probing What Different NLP Tasks Teach Machines about Function Word Comprehension](https://arxiv.org/abs/1904.11544) ("function word probing")
- [Investigating BERT’s Knowledge of Language: Five Analysis Methods with NPIs](https://arxiv.org/abs/1909.02597) ("BERT NPI paper")

To exactly reproduce experiments from [the ELMo's Friends paper](https://arxiv.org/abs/1812.10860) use the [`jsalt-experiments`](https://github.com/jsalt18-sentence-repl/jiant/tree/jsalt-experiments) branch. That will contain a snapshot of the code as of early August, potentially with updated documentation.

For the [edge probing paper](https://openreview.net/forum?id=SJzSgnRcKX) and the [BERT layer paper](https://arxiv.org/abs/1905.05950), see the [probing/](probing/) directory.

For the [function word probing paper](https://arxiv.org/abs/1904.11544), use [this branch](https://github.com/nyu-mll/jiant/tree/naacl_probingpaper) and refer to the instructions in the [scripts/fwords/](https://github.com/nyu-mll/jiant/tree/naacl_probingpaper/scripts/fwords) directory.

For the [BERT NPI paper](https://arxiv.org/abs/1909.02597) follow the instructions in [scripts/bert_npi](https://github.com/nyu-mll/jiant/tree/blimp-and-npi/scripts/bert_npi) on the [`blimp-and-npi`](https://github.com/nyu-mll/jiant/tree/blimp-and-npi) branch.

## Getting Help

Post an issue here on GitHub if you have any problems, and create a pull request if you make any improvements (substantial or cosmetic) to the code that you're willing to share.


## Contributing

We use the `black` coding style with a line limit of 100. After installing the requirements, simply running `pre-commit
install` should ensure you comply with this in all your future commits. If you're adding features or fixing a bug,
please also add the tests.

For any PR, make sure to update any existing `conf` files, tutorials, and scripts to match your changes. If your PR adds or changes functionality that can be directly tested, add or update a test.

For PRs that typical users will need to be aware of, include  make a matching PR to the [documentation](https://github.com/nyu-mll/jiant-site/edit/master/documentation/README.md). We will merge that documentation PR once the original PR is merged in _and pushed out in a release_. (Proposals for better ways to do this are welcome.)

For PRs that change package dependencies, update both `environment.yml` (used for conda) and `setup.py` (used by pip, and in automatic CircleCI tests).

## Releases

Releases are identified using git tags and distributed via PyPI for pip installation. After passing CI tests and creating a new git tag for a release, it can be uploaded to PyPI by running:

```bash
# create distribution
python setup.py sdist bdist_wheel

# upload to PyPI
python -m twine upload dist/*
```

More details can be found in [setup.py](setup.py).


## License

This package is released under the [MIT License](LICENSE.md). The material in the allennlp_mods directory is based on [AllenNLP](https://github.com/allenai/allennlp), which was originally released under the Apache 2.0 license.


## Acknowledgments

- Part of the development of `jiant` took at the 2018 Frederick Jelinek Memorial Summer Workshop on Speech and Language Technologies, and was supported by Johns Hopkins University with unrestricted gifts from Amazon, Facebook, Google, Microsoft and Mitsubishi Electric Research Laboratories.
- This work was made possible in part by a donation to NYU from Eric and Wendy Schmidt made
by recommendation of the Schmidt Futures program, and by support from Intuit Inc.
- We gratefully acknowledge the support of NVIDIA Corporation with the donation of a Titan V GPU used at NYU in this work.
- Developer Alex Wang is supported by the National Science Foundation Graduate Research Fellowship Program under Grant
No. DGE 1342536. Any opinions, findings, and conclusions or recommendations expressed in this
material are those of the author(s) and do not necessarily reflect the views of the National Science
Foundation.
- Developer Yada Pruksachatkun is supported by the Moore-Sloan Data Science Environment as part of the NYU Data Science Services initiative.
- Sam Bowman's work on `jiant` during Summer 2019 took place in his capacity as a visiting researcher at Google.
