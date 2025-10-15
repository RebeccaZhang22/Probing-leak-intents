#!/bin/bash

set -e 

model_path="/home/fit/qiuhan/.cache/huggingface/hub/Qwen2___5-7B-Instruct"
data_file_path="/home/fit/qiuhan/WORK/zyt/Probing-leak-intents-main/data/raw/Qwen2.5-7B-Instruct_t1_vllm_multiturn.csv"
output_dir="/home/fit/qiuhan/WORK/zyt/Probing-leak-intents-main/data/hiddens-multi_turn"
log_file_path="./0630-qwen-7b-hiddens-multi_turn.txt"

CUDA_VISIBLE_DEVICES=0 python ./scripts/extract_hiddens_multi_turn.py \
    --model_name_or_path "$model_path" \
    --file_path "$data_file_path" \
    --rougel_threshold 0.47 \
    --output_dir "$output_dir" \
    --batch_size 32 \
    --model_arch "Qwen2" \
    > "$log_file_path" 2>&1 &

wait 
echo "All extraction processess completed."
