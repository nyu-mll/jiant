python src/main.py --config config/defaults.conf --overrides "exp_name = bie, run_name = ptdb_defaults" --remote_log

python src/main.py --config config/defaults.conf --overrides "exp_name = dnc, run_name = recast_puns_elmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 0" --remote_log
