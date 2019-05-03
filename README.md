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

Follow this [getting started tutorial]()! 

## Official Documentation

Our official documentation is here: https://jiant.info/documentation#/
 

## Running
To run an experiment, make a config file similar to `config/demo.conf` with your model configuration. You can use the `--overrides` flag to override specific variables. For example:	
```sh	
python main.py --config_file config/demo.conf \
    --overrides "exp_name = my_exp, run_name = foobar, d_hid = 256"	
```	
will run the demo config, but output to `$JIANT_PROJECT_PREFIX/my_exp/foobar`.	
 To run the demo config, you will have to set environment variables. The best way to achieve that is to follow the instructions in [user_config_template.sh](user_config_template.sh)	
*  $JIANT_PROJECT_PREFIX: the where the outputs will be saved.	
*  $JIANT_DATA_DIR: location of the saved data. This is usually the location of the GLUE data in a simple default setup.	
*  $WORD_EMBS_FILE: location of any word embeddings you want to use (not necessary when using ELMo, GPT, or BERT). You can download GloVe (840B) [here](http://nlp.stanford.edu/data/glove.840B.300d.zip) or fastText (2M) [here](https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip).	
 To have `user_config.sh` run automatically, follow instructions in [scripts/export_from_bash.sh](export_from_bash.sh). 	


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

***It seems like my preproc/{task}\_\_{split}.data has nothing in it!***

This probably means that you probably ran the script before downloading the data for that task. Thus, delete the file from preproc and then run main.py again to build the data splits from scratch.

***How can I pass BERT embeddings straight to the classifier without a sentence encoder?***

Right now, you need to set `skip_embs=1` and `sep_embs_for_skip=1` just because of the current way 
our logic works. We're currently streamlining the logic around `sep_embs_for_skip` for the 1.0 release!


***How can I do STILTS-style training?***

Right now, we only support training in two stages. Training in more than two stages is possible, but will require you to divide your training up into multiple runs. For instance, assume you want to run multitask training on task set A, and then train on task set B, and finally fine-tune on task set C. You would perform the following:
- First run: pretrain on task set A
   - pretrain_tasks=“task_a1,task_a2”, target_tasks=“”
- Second run: load checkpoints, and train on task set B and then C:
   - load_model = 1
   - load_target_train_checkpoint_arg=/path/to/saved/run
   - pretrain_tasks=“task_b1,task_b2, target_tasks=task_c1,task_c2”



## Contributing

We use the `black` coding style with a line limit of 100. After installing the requirements, simply running `pre-commit
install` should ensure you comply with this in all your future commits. If you're adding features or fixing a bug,
please also add the tests.
