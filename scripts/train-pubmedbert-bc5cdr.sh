#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate biosyn

module load cudnn/7.6-cuda_9.0 cuda90

MODEL=biosyn-pubmedbert-bc5cdr
BERT_DIR=./pretrained/pt_pubmedbert
OUTPUT_DIR=./tmp/${MODEL}
DATA_DIR=./datasets/bc5cdr

python train.py \
    --model_dir ${BERT_DIR} \
    --train_dictionary_path ${DATA_DIR}/train_dictionary.txt \
    --train_dir ${DATA_DIR}/processed_train \
    --output_dir ${OUTPUT_DIR} \
    --use_cuda \
    --topk 20 \
    --epoch 10 \
    --train_batch_size 16\
    --initial_sparse_weight 0\
    --learning_rate 1e-5 \
    --max_length 25 \
    --dense_ratio 0.5
