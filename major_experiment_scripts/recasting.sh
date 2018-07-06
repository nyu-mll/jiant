'python src/main.py --config config/defaults.conf --overrides "train_task=recast-kg, exp_name = dnc, run_name = recast-kg_elmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-kg, exp_name = dnc, run_name = recast-kg_noelmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 1" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-kg, exp_name = dnc, run_name = recast-kg_elmo_hidropout, dropout = 0.4, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-kg, exp_name = dnc, run_name = recast-kg_noelmo_hidropout, dropout = 0.4, use_elmo_chars_only = 1" --remote_log'

'python src/main.py --config config/defaults.conf --overrides "train_task=recast-lexicosyntax, exp_name = dnc, run_name = recast-lexicosyntax_elmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-lexicosyntax, exp_name = dnc, run_name = recast-lexicosyntax_noelmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 1" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-lexicosyntax, exp_name = dnc, run_name = recast-lexicosyntax_elmo_hidropout, dropout = 0.4, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-lexicosyntax, exp_name = dnc, run_name = recast-lexicosyntax_noelmo_hidropout, dropout = 0.4, use_elmo_chars_only = 1" --remote_log'

'python src/main.py --config config/defaults.conf --overrides "train_task=recast-winogender, exp_name = dnc, run_name = recast-winogender_elmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-winogender, exp_name = dnc, run_name = recast-winogender_noelmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 1" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-winogender, exp_name = dnc, run_name = recast-winogender_elmo_hidropout, dropout = 0.4, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-winogender, exp_name = dnc, run_name = recast-winogender_noelmo_hidropout, dropout = 0.4, use_elmo_chars_only = 1" --remote_log'

'python src/main.py --config config/defaults.conf --overrides "train_task=recast-factuality, exp_name = dnc, run_name = recast-factuality_elmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-factuality, exp_name = dnc, run_name = recast-factuality_noelmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 1" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-factuality, exp_name = dnc, run_name = recast-factuality_elmo_hidropout, dropout = 0.4, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-factuality, exp_name = dnc, run_name = recast-factuality_noelmo_hidropout, dropout = 0.4, use_elmo_chars_only = 1" --remote_log'

'python src/main.py --config config/defaults.conf --overrides "train_task=recast-ner, exp_name = dnc, run_name = recast-ner_elmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-ner, exp_name = dnc, run_name = recast-ner_noelmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 1" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-ner, exp_name = dnc, run_name = recast-ner_elmo_hidropout, dropout = 0.4, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-ner, exp_name = dnc, run_name = recast-ner_noelmo_hidropout, dropout = 0.4, use_elmo_chars_only = 1" --remote_log'

'python src/main.py --config config/defaults.conf --overrides "train_task=recast-puns, exp_name = dnc, run_name = recast-puns_elmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-puns, exp_name = dnc, run_name = recast-puns_noelmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 1" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-puns, exp_name = dnc, run_name = recast-puns_elmo_hidropout, dropout = 0.4, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-puns, exp_name = dnc, run_name = recast-puns_noelmo_hidropout, dropout = 0.4, use_elmo_chars_only = 1" --remote_log'

'python src/main.py --config config/defaults.conf --overrides "train_task=recast-sentiment, exp_name = dnc, run_name = recast-sentiment_elmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-sentiment, exp_name = dnc, run_name = recast-sentiment_noelmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 1" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-sentiment, exp_name = dnc, run_name = recast-sentiment_elmo_hidropout, dropout = 0.4, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-sentiment, exp_name = dnc, run_name = recast-sentiment_noelmo_hidropout, dropout = 0.4, use_elmo_chars_only = 1" --remote_log'

'python src/main.py --config config/defaults.conf --overrides "train_task=recast-verbcorner, exp_name = dnc, run_name = recast-verbcorner_elmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-verbcorner, exp_name = dnc, run_name = recast-verbcorner_noelmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 1" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-verbcorner, exp_name = dnc, run_name = recast-verbcorner_elmo_hidropout, dropout = 0.4, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-verbcorner, exp_name = dnc, run_name = recast-verbcorner_noelmo_hidropout, dropout = 0.4, use_elmo_chars_only = 1" --remote_log'

'python src/main.py --config config/defaults.conf --overrides "train_task=recast-verbnet, exp_name = dnc, run_name = recast-verbnet_elmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-verbnet, exp_name = dnc, run_name = recast-verbnet_noelmo_lowdropout, dropout = 0.2, use_elmo_chars_only = 1" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-verbnet, exp_name = dnc, run_name = recast-verbnet_elmo_hidropout, dropout = 0.4, use_elmo_chars_only = 0" --remote_log'
'python src/main.py --config config/defaults.conf --overrides"train_task=recast-verbnet, exp_name = dnc, run_name = recast-verbnet_noelmo_hidropout, dropout = 0.4, use_elmo_chars_only = 1" --remote_log'

