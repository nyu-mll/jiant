python main.py --config ../config/defaults.conf --overrides "exp_name = base_rnn, sent_enc = rnn, d_hid = 1500, n_layers_enc = 2, skip_embs = 0"
python main.py --config ../config/defaults.conf --overrides "exp_name = small_rnn, sent_enc = rnn, d_hid = 1000, n_layers_enc = 2, skip_embs = 0"

python main.py --config ../config/defaults.conf --overrides "exp_name = base_transformer, sent_enc = transformer, d_hid = 768, n_layers_enc = 12, skip_embs = 0, n_heads = 12"
python main.py --config ../config/defaults.conf --overrides "exp_name = small_transformer, sent_enc = transformer, d_hid = 512, n_layers_enc = 8, skip_embs = 0, n_heads = 12"
python main.py --config ../config/defaults.conf --overrides "exp_name = smaller_transformer, sent_enc = transformer, d_hid = 384, n_layers_enc = 6, skip_embs = 0, n_heads = 12"

python main.py --config ../config/defaults.conf --overrides "exp_name = base_transformer_skip, sent_enc = transformer, d_hid = 768, n_layers_enc = 12, skip_embs = 1, n_heads = 12"
python main.py --config ../config/defaults.conf --overrides "exp_name = base_rnn_skip, sent_enc = rnn, d_hid = 1500, n_layers_enc = 2, skip_embs = 1"

python main.py --config ../config/defaults.conf --overrides "exp_name = base_transformer_high_lr, sent_enc = transformer, d_hid = 768, n_layers_enc = 12, skip_embs = 0, n_heads = 12, lr = 0.003"
python main.py --config ../config/defaults.conf --overrides "exp_name = base_transformer_low_lr, sent_enc = transformer, d_hid = 768, n_layers_enc = 12, skip_embs = 0, n_heads = 12, lr = 0.0003"
python main.py --config ../config/defaults.conf --overrides "exp_name = base_transformer_lower_lr, sent_enc = transformer, d_hid = 768, n_layers_enc = 12, skip_embs = 0, n_heads = 12, lr = 0.0001"

python main.py --config ../config/defaults.conf --overrides "exp_name = base_rnn_high_lr, sent_enc = rnn, d_hid = 1500, n_layers_enc = 2, skip_embs = 0, lr = 0.003"
python main.py --config ../config/defaults.conf --overrides "exp_name = base_rnn_low_lr, sent_enc = rnn, d_hid = 1500, n_layers_enc = 2, skip_embs = 0, lr = 0.0003"
python main.py --config ../config/defaults.conf --overrides "exp_name = base_rnn_lower_lr, sent_enc = rnn, d_hid = 1500, n_layers_enc = 2, skip_embs = 0, lr = 0.0001"
