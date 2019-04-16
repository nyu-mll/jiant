# The settings of the models, BiLSTM, Bow + GloVe, ELMo, OpenAI-GPT, BERT
Model_biLSTM = '''// Model, biLSTM
tokenizer = "MosesTokenizer" // The default tokenizer
sent_enc = "rnn"  // "bow", "rnn" for LSTM, "null"
bidirectional = 1
word_embs = "glove"
sep_embs_for_skip = 0 // Skip embedding uses the same embedder object as the original embedding (before skip)
elmo = 0
elmo_chars_only = 0
'''


Model_BERT = '''// Models, BERT
dropout = 0.1 // following BERT paper
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
elmo_chars_only = 0
openai_transformer = 0
'''


Model_elmo = '''// Model, elmo
tokenizer = "MosesTokenizer" // The default tokenizer
sent_enc = "rnn" // "bow", "rnn" for LSTM, "null"
bidirectional = 1
transfer_paradigm = "frozen" // "frozen" or "finetune"
elmo = 1  // If true, load and use ELMo.
elmo_chars_only = 0  // If true, use only the char CNN layer of ELMo. If false but elmo is true, use the full ELMo.
elmo_weight_file_path = none  // Path to ELMo RNN weights file.  Default ELMo weights will be used if "none".
'''

# The main body of the config file
Body = '''// Import the defaults using the relative path
include "../final.conf"


// Output path
project_dir = ${{JIANT_PROJECT_PREFIX}}


// Optimization
batch_size = 16
lr = 3e-4  // following "cola_elmo.conf"


// Target tasks
do_target_task_training = 1  // If true, after do_pretrain train the task-specific model parameters
write_preds = "val,test"  // 0 for none, or comma-separated splits in {{"train", "val", "test"}} 
                          // for which predictions are written to disk during do_full_eval


// Pretraining tasks
load_model = 0  // If true, restore from checkpoint when starting do_pretrain


// Model
{overridden_model_settings}'''



# Model Running Script Template
RunModelScr = '''python main.py --config_file {path_to_config_file} \\
    --overrides "exp_name = {overridden_exp_name}, run_name = {overridden_run_name}, target_tasks = "{overridden_target_tasks}", pretrain_tasks = "{overridden_pretrain_tasks}", do_pretrain = {overridden_do_pretrain}{overridden_allow_untrained}"'''


# Read-Eval-Pring-Loop Running Script Template
RunREPLScr = '''python cola_inference.py --config_file {path_to_params_conf} \\ 
--model_file_path {path_to_model_file}'''
