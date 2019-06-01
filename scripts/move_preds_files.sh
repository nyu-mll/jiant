FROM_DIR=$1
TO_DIR=$2

cp ${FROM_DIR}/appear/nli-alt_val.tsv ${TO_DIR}/appear.tsv
cp ${FROM_DIR}/compare/nli-alt_val.tsv ${TO_DIR}/compare.tsv
cp ${FROM_DIR}/factives/nli-alt_val.tsv ${TO_DIR}/factives.tsv
cp ${FROM_DIR}/implicatives/nli-alt_val.tsv ${TO_DIR}/implicatives.tsv
cp ${FROM_DIR}/negation/nli-prob-negation_val.tsv ${TO_DIR}/negation.tsv
cp ${FROM_DIR}/neutrals/nli-alt_val.tsv ${TO_DIR}/neutrals.tsv
cp ${FROM_DIR}/nps_final/nps_val.tsv ${TO_DIR}/nps.tsv
cp ${FROM_DIR}/prepswap/nli-prob-prepswap_val.tsv ${TO_DIR}/prepswap.tsv
cp ${FROM_DIR}/quant/nli-alt_val.tsv ${TO_DIR}/quant.tsv
cp ${FROM_DIR}/spatial/nli-alt_val.tsv ${TO_DIR}/spatial.tsv
