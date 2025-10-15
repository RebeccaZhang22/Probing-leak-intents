#!/bin/bash

set -e 

model_path="Qwen/Qwen2.5-7B-Instruct"
data_file_path="/home/fit/qiuhan/WORK/zyt/Probing-leak-intents-main/data/pleak_t1_vllm.csv"
output_dir="/home/fit/qiuhan/WORK/zyt/Probing-leak-intents-main/data/hiddens-pleak"
log_file_path="./0630-qwen-7b-hiddens.txt"

CUDA_VISIBLE_DEVICES=0 python ./scripts/extract_hiddens.py \
    --model_name_or_path "$model_path" \
    --file_path "$data_file_path" \
    --rougel_threshold 0.47 \
    --output_dir "$output_dir" \
    --batch_size 32 \
    --model_arch "Qwen2" \
    > "$log_file_path" 2>&1 &

wait 
echo "All extraction processess completed."
