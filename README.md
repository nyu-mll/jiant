# jiant

[![CircleCI](https://circleci.com/gh/nyu-mll/jiant/tree/master.svg?style=svg)](https://circleci.com/gh/nyu-mll/jiant/tree/master)

`jiant` is a work-in-progress software toolkit for natural language processing research, designed to facilitate work on multitask learning and transfer learning for sentence understanding tasks.

A few things you might want to know about `jiant`:

- `jiant` is configuration-driven. You can run an enormous variety of experiments by simply writing configuration files. Of course, if you need to add any major new features, you can also easily edit or extend the code.
- `jiant` contains implementations of strong baselines for the [GLUE](https://gluebenchmark.com) and [SuperGLUE](https://super.gluebenchmark.com/) benchmarks, and it's the recommended starting point for work on these benchmarks.
- `jiant` was developed at [the 2018 JSALT Workshop](https://www.clsp.jhu.edu/workshops/18-workshop/) by [the General-Purpose Sentence Representation Learning](https://jsalt18-sentence-repl.github.io/) team and is maintained by [the NYU Machine Learning for Language Lab](https://wp.nyu.edu/ml2/people/), with help from [many outside collaborators](https://github.com/nyu-mll/jiant/graphs/contributors) (especially Google AI Language's [Ian Tenney](https://ai.google/research/people/IanTenney)).
- `jiant` is built on [PyTorch](https://pytorch.org). It also uses many components from [AllenNLP](https://github.com/allenai/allennlp) and HuggingFace PyTorch [implementations](https://github.com/huggingface/pytorch-pretrained-BERT) of BERT and GPT.
- The name `jiant` doesn't mean much. The 'j' stands for JSALT. That's all the acronym we have.

## Getting Started

To run a simple example demo experiment using data from GLUE:

1. Clone `jiant` and its submodules.
2. Follow this [Get Started tutorial]()! 

### Submodules

This project uses [git submodules](https://blog.github.com/2016-02-01-working-with-submodules/) to manage some dependencies on other research code, in particular for loading CoVe and the OpenAI transformer model. In order to make sure you get these repos when you download `jiant/`, add `--recursive` to your clone command:

```sh
git clone --recursive git@github.com:jsalt18-sentence-repl/jiant.git jiant
```

If you already cloned and just need to get the submodules, you can run:
```sh
git submodule update --init --recursive
```
## Official Documentation

Our official documentation is here: https://jiant.info/documentation#/
 
## Suggested Citation

If you use `jiant` in academic work, please cite it directly:

```
@misc{wang2019jiant,
    author = {Alex Wang and Ian F. Tenney and Yada Pruksachatkun and Katherin Yu and Jan Hula and Patrick Xia and Raghu Pappagari and Shuning Jin and R. Thomas McCoy and Roma Patel and Yinghui Huang and Jason Phang and Edouard Grave and Najoung Kim and Phu Mon Htut and Thibault F'{e}vry and Berlin Chen and Nikita Nangia and Haokun Liu and and Anhad Mohananey and Shikha Bordia and Ellie Pavlick and Samuel R. Bowman},
    title = {{jiant} 0.9: A software toolkit for research on general-purpose text understanding models},
    howpublished = {\url{http://jiant.info/}},
    year = {2019}
}
```

## Papers

`jiant` has been used in these three papers so far:

- [Looking for ELMo's Friends: Sentence-Level Pretraining Beyond Language Modeling](https://arxiv.org/abs/1812.10860)
- [What do you learn from context? Probing for sentence structure in contextualized word representations](https://openreview.net/forum?id=SJzSgnRcKX) ("Edge Probing")
- [Probing What Different NLP Tasks Teach Machines about Function Word Comprehension](https://arxiv.org/abs/1904.11544)

To exactly reproduce experiments from [the ELMo's Friends paper](https://arxiv.org/abs/1812.10860) use the [`jsalt-experiments`](https://github.com/jsalt18-sentence-repl/jiant/tree/jsalt-experiments) branch. That will contain a snapshot of the code as of early August, potentially with updated documentation.

For the [edge probing paper](https://openreview.net/forum?id=SJzSgnRcKX), see the [probing/](probing/) directory.


## License

This package is released under the [MIT License](LICENSE.md). The material in the allennlp_mods directory is based on [AllenNLP](https://github.com/allenai/allennlp), which was originally released under the Apache 2.0 license.

## Getting Help

Post an issue here on GitHub if you have any problems, and create a pull request if you make any improvements (substantial or cosmetic) to the code that you're willing to share.

## FAQs

It seems like my preproc/{task}\_\_{split}.data has nothing in it!

This probably means that you probably ran the script before downloading the data for that task. Thus, delete the file from preproc and then run main.py again to build the data splits from scratch.

## Contributing

We use the `black` coding style with a line limit of 100. After installing the requirements, simply running `pre-commit
install` should ensure you comply with this in all your future commits. If you're adding features or fixing a bug,
please also add the tests.
