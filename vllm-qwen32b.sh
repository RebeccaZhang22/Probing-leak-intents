#!/bin/bash

# model_dir="/PATH/TO/YOUR/LOCAL/QWEN/32B"

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-32B-Instruct" \
    --dtype auto \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --enforce_eager \
    --gpu-memory-utilization "0.99" \
    --enable-prefix-caching \
    --port 8000 \
    --guided-decoding-backend "lm-format-enforcer" \
    --max-model-len 9000 \
    --disable-log-requests
