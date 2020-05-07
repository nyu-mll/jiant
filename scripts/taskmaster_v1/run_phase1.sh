#!/bin/bash

. ./transfer_analysis.sh --source-only

for task in  "sst" "SocialIQA" "qqp" "mnli" "scitail" "qasrl" "qamr" "squad" "cosmosqa" "hellaswag" "commonsenseqa"
do
  ez_first_intermediate_exp 1111001 $task
  ez_first_intermediate_exp 921 $task
  ez_first_intermediate_exp 523821  $task
end
