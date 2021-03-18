#!/bin/bash

# Check for debug flag
debug_flag=""
while getopts ":d" opt; do
  case $opt in
    d)
      echo "DEBUG_MODE=ON" >&2
      debug_flag="--debug_mode"
      ;;
  esac
done

eval "$(conda shell.bash hook)"
conda activate biosyn

module load cudnn/7.6-cuda_9.0 cuda90

MODEL=biosyn-bc5cdr-disease
MODEL_DIR=./tmp/${MODEL}
OUTPUT_DIR=./tmp/${MODEL}
DATA_DIR=./datasets/bc5cdr-disease

python eval.py \
    --model_dir ${MODEL_DIR} \
    --dictionary_path ${DATA_DIR}/test_dictionary.txt \
    --data_dir ${DATA_DIR}/processed_test \
    --output_dir ${OUTPUT_DIR} \
    --use_cluster_linking \
    --use_cuda \
    --topk 16 \
    --max_length 25 \
    --save_predictions $debug_flag
