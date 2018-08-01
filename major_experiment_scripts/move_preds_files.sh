FROM_DIR=$1 #"/nfs/jsalt/exp/ellie-k2/probing_ccglogreg_fullmnli/"
TO_DIR=$2 #"/nfs/jsalt/share/models_to_probe/results_logreg_fullmnli/ccg"

sudo cp ${FROM_DIR}/appear/nli-alt_val.tsv ${TO_DIR}/appear.tsv
sudo cp ${FROM_DIR}/compare/nli-alt_val.tsv ${TO_DIR}/compare.tsv
sudo cp ${FROM_DIR}/factives/nli-alt_val.tsv ${TO_DIR}/factives.tsv
sudo cp ${FROM_DIR}/implicatives/nli-alt_val.tsv ${TO_DIR}/implicatives.tsv
sudo cp ${FROM_DIR}/negation/nli-prob-negation_val.tsv ${TO_DIR}/negation.tsv
sudo cp ${FROM_DIR}/neutrals/nli-alt_val.tsv ${TO_DIR}/neutrals.tsv
sudo cp ${FROM_DIR}/nps_final/nps_val.tsv ${TO_DIR}/nps.tsv
sudo cp ${FROM_DIR}/prepswap/nli-prob-prepswap_val.tsv ${TO_DIR}/prepswap.tsv
sudo cp ${FROM_DIR}/quant/nli-alt_val.tsv ${TO_DIR}/quant.tsv
sudo cp ${FROM_DIR}/spatial/nli-alt_val.tsv ${TO_DIR}/spatial.tsv
