python src/main.py --config config/defaults.conf --overrides "exp_name = dnc, run_name = recast_puns_elmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 0" --remote_log
python src/main.py --config config/defaults.conf --overrides "exp_name = dnc, run_name = recast_puns_noelmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 1" --remote_log
python src/main.py --config config/defaults.conf --overrides "exp_name = dnc, run_name = recast_puns_elmo_hidropout, dropout = 0.4, use_elmo_chars_only = 0" --remote_log
python src/main.py --config config/defaults.conf --overrides "exp_name = dnc, run_name = recast_puns_noelmo_hidropout, dropout = 0.4, use_elmo_chars_only = 1" --remote_log
