#!/bin/bash

model_dir="/home/fit/qiuhan/.cache/huggingface/hub/Qwen2___5-7B-Instruct"

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --dtype auto \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --enforce_eager \
    --gpu-memory-utilization "0.95" \
    --enable-prefix-caching \
    --port 8000 \
    --guided-decoding-backend "lm-format-enforcer" \
    --max-model-len 8000
