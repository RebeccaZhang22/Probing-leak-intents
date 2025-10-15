#!/bin/bash

DATASET_DIR="/home/fit/qiuhan/WORK/zyt/Probing-leak-intents-main/data/multi-turn-dataset"
OUTPUT_DIR="/home/fit/qiuhan/WORK/zyt/Probing-leak-intents-main/data/multi-turn-hybrid-relabeled"

STRATEGY="hybrid"  # "rougel" or "hybrid" or "llm"
ROUGEL_THRESHOLD=0.46

LABEL_FILE="/home/fit/qiuhan/WORK/zyt/Probing-leak-intents-main/data/label/Qwen2.5-7B-Instruct_t1_vllm_multiturn.csv"
MODEL_ARCH="qwen2"  # "qwen2" or "llama3"

if [ "$STRATEGY" == "rougel" ]; then
    if [ -z "$ROUGEL_THRESHOLD" ]; then
        echo "Error: rougel_threshold is required for rougel strategy"
        exit 1
    fi
    python scripts/relabel_datasets.py \
        --dataset_dir "$DATASET_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --labeling_strategy rougel \
        --rougel_threshold "$ROUGEL_THRESHOLD" \
        --model_arch "$MODEL_ARCH"
elif [ "$STRATEGY" == "hybrid" ]; then
    echo "Using hybrid strategy"
    python scripts/relabel_datasets_multi_turn.py \
        --dataset_dir "$DATASET_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --labeling_strategy hybrid \
        --label_file "$LABEL_FILE" \
        --model_arch "$MODEL_ARCH"
elif [ "$STRATEGY" == "llm" ]; then
    echo "Using LLM strategy"
    python scripts/relabel_datasets.py \
        --dataset_dir "$DATASET_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --labeling_strategy llm \
        --label_file "$LABEL_FILE" \
        --model_arch "$MODEL_ARCH"
else
    echo "Error: strategy must be either 'rougel', 'hybrid' or 'llm'"
    exit 1
fi
