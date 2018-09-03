#!/bin/bash

pushd $(dirname $0)
python ccg_proc.py
python ccg_to_num.py
python moses_aligner.py ccg_num.train
python moses_aligner.py ccg_num.dev
python moses_aligner.py ccg_num.test
python zipper.py ccg_num.train ccg_num.train.moses
python zipper.py ccg_num.dev ccg_num.dev.moses
python zipper.py ccg_num.test ccg_num.test.moses
mv ccg_num.train.zipped ccg_1363.train
mv ccg_num.dev.zipped ccg_1363.dev
mv ccg_num.test.zipped ccg_1363.test


