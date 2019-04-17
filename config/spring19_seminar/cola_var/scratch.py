# The settings of the models, BiLSTM, Bow + GloVe, ELMo, OpenAI-GPT, BERT
Model_biLSTM = '''// Model, biLSTM
tokenizer = "MosesTokenizer" // The default tokenizer
sent_enc = "rnn"  // "bow", "rnn" for LSTM, "null"
bidirectional = 1
word_embs = "glove"
sep_embs_for_skip = 0 // Skip embedding uses the same embedder object as the original embedding (before skip)
elmo = 0
elmo_chars_only = 0'''


Model_BERT = '''// Model, BERT
tokenizer = "bert-large-cased"
sent_enc = "null" // "bow", "rnn" for LSTM, "null"
transfer_paradigm = "finetune" // "frozen" or "finetune"
bert_fine_tune = 1
bert_model_name = "bert-large-cased"  // If nonempty, use this BERT model for representations.
                                        // Available values: bert-base-uncased, bert-large-cased, ...
bert_embeddings_mode = "none"  // How to handle the embedding layer of the BERT model:
                               // "none" for only top-layer activation,
sep_embs_for_skip = 1 // Skip embedding uses the same embedder object as the original embedding (before skip)                               
elmo = 0
elmo_chars_only = 0'''


Model_bow_glove = '''// Model, bow + GloVe
tokenizer = "MosesTokenizer" // The default tokenizer
sent_enc = "bow" // "bow", "rnn" for LSTM, "null"
word_embs = "glove"
elmo = 0
elmo_chars_only = 0'''


# The main body of the config file
Body = '''// Import the defaults using the relative path
include "../final.conf"


// Output path
project_dir = ${{JIANT_PROJECT_PREFIX}}


// Optimization
batch_size = 16
dropout = 0.1 // following BERT paper
lr = 2e-5  // following Jason, Alex


// Target tasks
do_target_task_training = 1  // If true, after do_pretrain train the task-specific model parameters
write_preds = "val,test"  // 0 for none, or comma-separated splits in {{"train", "val", "test"}} 
                          // for which predictions are written to disk during do_full_eval


// Pretraining tasks
load_model = 0  // If true, restore from checkpoint when starting do_pretrain


// Models
{overridden_model_settings}'''


# Model Running Script Template
RunModelHead = '''#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --job-name=myrun
#SBATCH --output=slurm_%j.out
'''

RunModelScr = '''python main.py --config_file {path_to_config_file} \\
    --overrides "exp_name = {overridden_exp_name}, run_name = {overridden_run_name}, target_tasks = "{overridden_target_tasks}", pretrain_tasks = "{overridden_pretrain_tasks}", do_pretrain = {overridden_do_pretrain}{overridden_allow_untrained}"'''


# Read-Eval-Pring-Loop Running Script Template
RunREPLScr = '''python cola_inference.py --config_file {path_to_params_conf} \\ 
--model_file_path {path_to_model_file}'''
