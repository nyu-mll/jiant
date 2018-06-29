python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, run_name = take2_base_rnn, sent_enc = rnn, d_hid = 1500, n_layers_enc = 2"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, run_name = take2_base_rnn_elmo, sent_enc = rnn, d_hid = 1500, n_layers_enc = 2, elmo_chars_only = 0"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, run_name = take2_base_rnn_deep, sent_enc = rnn, d_hid = 1500, n_layers_enc = 3"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, run_name = take2_base_rnn_elmo_deep, sent_enc = rnn, d_hid = 1500, n_layers_enc = 3, elmo_chars_only = 0"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, run_name = take2_base_rnn_small_attn, sent_enc = rnn, d_hid = 1500, n_layers_enc = 2, d_hid_attn = 256"
python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, run_name = take2_base_rnn_big_attn, sent_enc = rnn, d_hid = 1500, n_layers_enc = 2, d_hid_attn = 1024"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, run_name = take2_base_rnn_small_proj, sent_enc = rnn, d_hid = 1500, n_layers_enc = 2, d_proj = 256"
python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-2, run_name = take2_base_rnn_big_proj, sent_enc = rnn, d_hid = 1500, n_layers_enc = 2, d_proj = 1024"


python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-3, run_name = take2_base_transformer_highwarm, sent_enc = transformer, d_hid = 768, n_layers_enc = 12, n_heads = 8, warmup = 10000"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-3, run_name = take2_base_transformer, sent_enc = transformer, d_hid = 768, n_layers_enc = 12, n_heads = 8, warmup = 4000"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-3, run_name = take2_base_transformer_lowlr, sent_enc = transformer, d_hid = 768, n_layers_enc = 12, n_heads = 8, warmup = 4000, lr = 0.00003"

python main.py --config ../config/defaults.conf --overrides "exp_name = common-indexed-tasks-3, run_name = take2_base_transformer_lowwarm, sent_enc = transformer, d_hid = 768, n_layers_enc = 12, n_heads = 8, warmup = 1000"

