#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

lang=$1 #the language, 'en' for English wikipedia
export BERT_PREP_WORKING_DIR=$2

# clone wikiextractor if it doesn't exist
if [ ! -d "wikiextractor" ]; then
    git clone https://github.com/attardi/wikiextractor.git
fi

echo "Downloading $lang wikpedia in directory $save_dir"
# Download
python3 bertPrep.py --action download --dataset wikicorpus_$lang


# Properly format the text files
python3 bertPrep.py --action text_formatting --dataset wikicorpus_$lang


# Shard the text files (group wiki+books then shard)
python3 bertPrep.py --action sharding --dataset wikicorpus_$lang


# Combine sharded files into one
save_dir=$BERT_PREP_WORKING_DIR/sharded_training_shards_256_test_shards_256_fraction_0.2/wikicorpus_$lang
cat $save_dir/*training*.txt > $save_dir/train_$lang.txt
cat $save_dir/*test*.txt > $save_dir/test_$lang.txt
rm -rf $save_dir/wiki*training*.txt
rm -rf $save_dir/wiki*test*.txt

# remove some remaining xml tags
sed -i 's/<[^>]*>//g' $save_dir/train_$lang.txt
sed -i 's/<[^>]*>//g' $save_dir/test_$lang.txt

echo "Your corpus is saved in $save_dir"

