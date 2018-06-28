python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_rnn, sent_enc = rnn, d_hid = 1000, n_layers_enc = 2, skip_embs = 0, batch_size = 8"
python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = small_rnn, sent_enc = rnn, d_hid = 750, n_layers_enc = 2, skip_embs = 0, batch_size = 8"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_transformer, sent_enc = transformer, d_hid = 768, n_layers_enc = 6, skip_embs = 0, n_heads = 6, batch_size = 8"
python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = small_transformer, sent_enc = transformer, d_hid = 512, n_layers_enc = 6, skip_embs = 0, n_heads = 6, batch_size = 8"
python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = smaller_transformer, sent_enc = transformer, d_hid = 384, n_layers_enc = 6, skip_embs = 0, n_heads = 6, batch_size = 8"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_transformer_skip, sent_enc = transformer, d_hid = 768, n_layers_enc = 6, skip_embs = 1, n_heads = 8, batch_size = 8"	
python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_rnn_skip, sent_enc = rnn, d_hid = 1000, n_layers_enc = 2, skip_embs = 1, batch_size = 8"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_transformer_higher_lr, sent_enc = transformer, d_hid = 768, n_layers_enc = 6, skip_embs = 0, n_heads = 8, lr = 0.001"
python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_transformer_high_lr, sent_enc = transformer, d_hid = 768, n_layers_enc = 6, skip_embs = 0, n_heads = 8, lr = 0.0003"
python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_transformer_low_lr, sent_enc = transformer, d_hid = 768, n_layers_enc = 6, skip_embs = 0, n_heads = 8, lr = 0.00003"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_rnn_higher_lr, sent_enc = rnn, d_hid = 1000, n_layers_enc = 2, skip_embs = 0, lr = 0.001"
python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_rnn_high_lr, sent_enc = rnn, d_hid = 1000, n_layers_enc = 2, skip_embs = 0, lr = 0.0003"
python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_rnn_low_lr, sent_enc = rnn, d_hid = 1000, n_layers_enc = 2, skip_embs = 0, lr = 0.00003"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_rnn_elmo, sent_enc = rnn, d_hid = 1000, n_layers_enc = 2, skip_embs = 0, elmo_chars_only = 0"
python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_transformer_elmo, sent_enc = transformer, d_hid = 768, n_layers_enc = 6, skip_embs = 0, n_heads = 8, elmo_chars_only = 0"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_transformer_skip_elmo, sent_enc = transformer, d_hid = 768, n_layers_enc = 6, skip_embs = 1, n_heads = 8, elmo_chars_only = 0"
python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, eval_tasks = \"sst,qnli,rte\", run_name = base_rnn_skip_elmo, sent_enc = rnn, d_hid = 1000, n_layers_enc = 2, skip_embs = 1, elmo_chars_only = 0"

