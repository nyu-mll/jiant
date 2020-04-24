# Script used to split new NLI-format data files.

export BASENAME="F-main"

head -n 1 "$BASENAME.tsv" > "$BASENAME-header.tsv"
tail -n +2 "$BASENAME.tsv" > "$BASENAME-body.tsv"
sort -R "$BASENAME-body.tsv" > "$BASENAME-shuf.tsv"
head -n 8500 "$BASENAME-shuf.tsv" > "$BASENAME-train-body.tsv"
tail -n +8501 "$BASENAME-shuf.tsv" > "$BASENAME-val-body.tsv"
cat "$BASENAME-header.tsv" "$BASENAME-train-body.tsv" > "$BASENAME-train.tsv"
cat "$BASENAME-header.tsv" "$BASENAME-val-body.tsv" > "$BASENAME-val.tsv"

mkdir "$BASENAME"
cp "$BASENAME-train.tsv" "$BASENAME/train.tsv"

# Hack to make it possible to use an MNLI data-reader exactly as-is:
# Duplicate the validation set four times.
cp "$BASENAME-val.tsv" "$BASENAME/dev_matched.tsv"
cp "$BASENAME-val.tsv" "$BASENAME/dev_mismatched.tsv"
cp "$BASENAME-val.tsv" "$BASENAME/test_matched.tsv"
cp "$BASENAME-val.tsv" "$BASENAME/test_mismatched.tsv"
