
#!/bin/bash

TARGET_DIR="/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/"

THIS_DIR="/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/probing/data/"

set -e
if [ ! -d $TARGET_DIR ]; then
  mkdir $TARGET_DIR
fi

function fetch_data() {
  mkdir -p $TARGET_DIR/raw
  pushd $TARGET_DIR/raw

  git clone https://github.com/google-research-datasets/gap-coreference.git

  popd
}



# Convert GAP to edge probing JSON format.
for split in "gap-development" "gap-test" "gap-validation"; do
    python $THIS_DIR/convert-gap.py -i "/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/${split}.tsv" \
        -o "/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/${split}.json"
done

# Print dataset stats for sanity-check.
python ${THIS_DIR%jiant*}/jiant/probing/edge_data_stats.py -i $TARGET_DIR/*.json
