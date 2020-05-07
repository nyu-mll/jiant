## Instructions to run Taskmaster Experiments

This code allows for experiments in the paper [Intermediate-Task Transfer Learning with Pretrained Models for Natural Language Understanding: When and Why Does It Work?](https://arxiv.org/abs/2005.00628).
To run the code for this project, first please follow the instructions from the main README.md to set up 
`jiant`. 

Our main experiment setup is in `transfer_analsysis.sh`, including functions to build and launch command line experiments and the specific hyperparameters we use for each task (after hyperparameter search). Our 
hyperparameter search functions are also defined in that file. 

### Running Experiments 
To first run experiments on the intermediate tasks, run `./run_phase1.sh`. 

Then, to do the intermediate to target tasks, we provide a `ez_run_intermediate_to_target_task` command defined in `transfer_analysis.sh`, which takes in the run_name, intermediate_task name, target_task name, and the path to the project 
directory. As an example, to train on the BoolQ task with RoBERTa pretrained on the CosmosQA intermediate task
(with checkpoints saved in `/x/y/roberta-large/cosmosqa_run921`) on run 111011, we would call `ez_run_intermediate_to_target_task 111011 cosmosqa boolq /x/y/roberta-large/cosmosqa_run921`.

Similarly, we provide a function to train on the probing tasks with `ez_run_intermediate_to_probing`, which takes in the run_name, intermediate_task name, probing_task name, and the path to the project. 

We organize our experiments as such (which is what the `transfer_analysis.sh` code will automatically create)
* Our intermediate task experiments are saved in a `roberta-large` experiment directory within the project directory. 
* Then, our intermediate to transfer experiments are saved such that for intermediate task A and target task B (in run C), we 
save that particular run in the experiment directory `A`, under run name `B_runC`. 

### Aggregating Results
In each of the experiment directories for the intermediate tasks, there should be a `results.tsv` file (for example, for all experiments 
with CosmosQA as an intermediate task, you would find this file in /path/to/project/directory/cosmosqa/results.tsv). This file writes out 
the results of each of your runs. Make sure that all of the runs have finished before you proceed to this step. 

Move the contents of `results.tsv` for each intermediate task to a directory with the below structure. 

This will generate two pandas dataframes, one with intermediate to target task results and the other 
with intermediate to probing task results. 

## Generating Correlations
Then, after we have the two pandas dataframes, run `calculate_correlations.py`. This will generate the graph of correlations (Figure 3)
in the paper. 



 