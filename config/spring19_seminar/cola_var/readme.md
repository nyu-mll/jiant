## Config files for experiments using cola(-like) target tasks:

The target tasks of these files are all `cola`.
You can run them directly without setting `exp_name` and `run_name`, 
which are set to `exp_pretrain` and `run`_`[model name]`_`[pretraining tasks]`
The config file is named in the format `cola`_`[model name]`_`[pretraining tasks]`.`conf`.

Config files that have been tested:
* `cola_bert_mnli.conf` (`batch_size` might need be overrided to a smaller value)
* `cola_bert_none.conf`
* `cola_bert_sst.conf`
* `cola_bilstm_none.conf` (using from-"scratch" as word embedding)
* `cola_bilstm_sst.conf`
* `cola_bilstm_mnli.conf`

Config files yet to be tested:
all files containing `ccg` as pretraining tasks
