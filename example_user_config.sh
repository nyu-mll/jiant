# machine-specific paths
# This is an example. Replace it with your own local setup,
# remove the 'example' from the filename, and run experiments
# with: 
# source ../user_config.sh; python main.py --config ../config/demo.conf --overrides "do_train = 0"

export JIANT_PROJECT_PREFIX=/Users/Bowman/Drive/JSALT
export JIANT_DATA_DIR=/Users/Bowman/Drive/JSALT/jiant/glue_data
export WORD_EMBS_FILE=~/glove.840B.300d.txt
export FASTTEXT_MODEL_FILE=None
export FASTTEXT_EMBS_FILE=None

echo "Loaded Sam's config."
