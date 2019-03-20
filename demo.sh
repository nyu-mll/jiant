#

# Quick-start: run this

export JIANT_PROJECT_PREFIX=coreference_exp
export JIANT_DATA_DIR=/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data
export NFS_PROJECT_PREFIX=/nfs/jsalt/exp/nkim
export NFS_DATA_DIR=/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant
export WORD_EMBS_FILE=/misc/vlgscratch4/BowmanGroup/yp913/jiant/data/glove.840B.300d.txt
export FASTTEXT_MODEL_FILE=None
export FASTTEXT_EMBS_FILE=None
module load anaconda3
module load cuda 10.0
source activate jiant_new
python main.py --config_file config/benchmarkssuperglue/gap_bert.conf

