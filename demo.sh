export JIANT_PROJECT_PREFIX=openai_exp
export JIANT_DATA_DIR=/Users/yadapruksachatkun/coref-jiant/data
export NFS_PROJECT_PREFIX=/nfs/jsalt/exp/nkim
export NFS_DATA_DIR=/Users/yadapruksachatkun/coref-jiant/data
export WORD_EMBS_FILE=/misc/vlgscratch4/BowmanGroup/yp913/jiant/data/glove.840B.300d.txt
export FASTTEXT_MODEL_FILE=None
export FASTTEXT_EMBS_FILE=None
python main.py --config_file config/edgeprobe_openai.conf
