#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

hiddens_path="/home/fit/qiuhan/WORK/zyt/Probing-leak-intents-main/data/hiddens-multi_turn/rougel_0.47_features.pt"
SEED=42

python scripts/organize_dataset.py --input ${hiddens_path} \
    --output_dir "/home/fit/qiuhan/WORK/zyt/Probing-leak-intents-main/data/multi-turn-dataset" \
    --val_ratio 0.0 \
    --test_ratio 0.0 \
    --seed ${SEED} \
