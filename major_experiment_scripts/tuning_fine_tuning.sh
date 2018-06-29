# Train once, fine-tune a bunch of ways on a bunch of things.
python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, train_tasks = mnli-fiction, run_name = setup, elmo_chars_only = 0, dropout = 0.4, val_interval = 500, do_eval = false"

# TODO: Set up checkpoints
python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, run_name = qqp_st, train_tasks = mnli-fiction, eval_tasks = qqp, do_train = false, load_model = true, qqp_d_hid_attn = 512"
python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, run_name = qqp_lg, train_tasks = mnli-fiction, eval_tasks = qqp, do_train = false, load_model = true, qqp_d_hid_attn = 768"
python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, run_name = qqp_sm, train_tasks = mnli-fiction, eval_tasks = qqp, do_train = false, load_model = true, qqp_d_hid_attn = 256"
python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, run_name = qqp_xs, train_tasks = mnli-fiction, eval_tasks = qqp, do_train = false, load_model = true, qqp_d_hid_attn = 128"

python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, run_name = mrpc_st, train_tasks = mnli-fiction, eval_tasks = mrpc, do_train = false, load_model = true, mrpc_d_proj = 256"
python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, run_name = mrpc_lg, train_tasks = mnli-fiction, eval_tasks = mrpc, do_train = false, load_model = true, mrpc_d_proj = 512"
python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, run_name = mrpc_sm, train_tasks = mnli-fiction, eval_tasks = mrpc, do_train = false, load_model = true, mrpc_d_proj = 128"

python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, run_name = wnli_st, train_tasks = mnli-fiction, eval_tasks = wnli, do_train = false, load_model = true, wnli_classifier_hid_dim = 128"
python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, run_name = wnli_sm, train_tasks = mnli-fiction, eval_tasks = wnli, do_train = false, load_model = true, wnli_classifier_hid_dim = 64"
python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, run_name = wnli_xs, train_tasks = mnli-fiction, eval_tasks = wnli, do_train = false, load_model = true, wnli_classifier_hid_dim = 32"

python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, run_name = cola_st, train_tasks = mnli-fiction, eval_tasks = cola, do_train = false, load_model = true, cola_classifier_hid_dim = 512"
python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, run_name = cola_lg, train_tasks = mnli-fiction, eval_tasks = cola, do_train = false, load_model = true, cola_classifier_hid_dim = 1028"
python main.py --config ../config/defaults.conf --overrides "exp_name = tuning_fine_tuning, run_name = cola_sm, train_tasks = mnli-fiction, eval_tasks = cola, do_train = false, load_model = true, cola_classifier_hid_dim = 256"

