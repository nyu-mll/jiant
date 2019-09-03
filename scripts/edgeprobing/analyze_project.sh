#!/bin/bash

# Convenience script to analyze a directory ("project") of experiments.
#
# Usage: ./analyze_project.sh /path/to/project
#
# Expected directory structure:
# <project_dir>/
#   <experiment_1>/
#     run/
#     vocab/
#   <experiment_2>/
#     run/
#     vocab/
#   (...)
#
# Will create <project_dir>/scores.tsv and <project_dir>/scalars.tsv as output.

PROJECT_DIR=$1

set -eu

if [ -z $PROJECT_DIR ]; then
    echo "You must provide a project directory!"
    exit 1
fi

THIS_DIR=$(realpath $(dirname $0))
JIANT_DIR=${THIS_DIR%jiant*}/jiant

all_runs=( ${PROJECT_DIR}/*/run* )
echo "Found runs:"
for run in "${all_runs[@]}"; do
  echo $run
done

nproc=$( grep -c ^processor /proc/cpuinfo )
nparallel=$( expr $nproc - 1 )

set -x

python $JIANT_DIR/probing/analyze_runs.py \
    -i "${all_runs[@]}" -o ${PROJECT_DIR}/scores.tsv \
    --parallel $(( $nparallel > 10 ? 10 : $nparallel ))

python $JIANT_DIR/probing/get_scalar_mix.py \
    -i "${all_runs[@]}" -o ${PROJECT_DIR}/scalars.tsv

