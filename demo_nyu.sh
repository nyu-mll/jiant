export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_DATA_DIR="/Users/yadapruksachatkun/jiant_new/data/"
export NFS_PROJECT_PREFIX="/nfs/jsalt/exp/nkim" 
export NFS_DATA_DIR="/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant"
export WORD_EMBS_FILE="/misc/vlgscratch4/BowmanGroup/yp913/jiant/data/glove.840B.300d.txt"
export FASTTEXT_MODEL_FILE=None 
export FASTTEXT_EMBS_FILE=None  
source activate jiant

#function rte() {
#    python main.py --config config/defaults.conf --overrides "run_name = rte_glue, reload_tasks=1, 	pretrain_tasks = \"rte\", target_tasks = \"glue-diagnostic\", sep_embs_for_skip = 1, elmo = 1, do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 625"
#}
python main.py --config config/superglue-bow.conf --overrides "exp_name = \"bow-rte\", run_name = rte, pretrain_tasks = \"rte-superglue\", target_tasks = \"rte-superglue,	winogender-diagnostic\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, val_interval = 625"
python main.py --config config/superglue-bow.conf --overrides "exp_name = \"bow-wsc\", run_name = wsc, pretrain_tasks = \"winograd-coreference\", target_tasks = \"winograd-coreference\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, val_interval = 139, optimizer = adam"