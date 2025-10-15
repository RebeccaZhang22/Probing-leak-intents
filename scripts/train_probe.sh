#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

dataset_dir="/PATH/TO/YOUR/DATASET/DIR"

python scripts/train_probe.py \
    --probe_type "logistic_regression" \
    --train_data_path "${dataset_dir}/train.pt" \
    --val_data_path "${dataset_dir}/val.pt" \
    --heldout_attacks_data_path "${dataset_dir}/heldout_attacks.pt" \
    --heldout_strict_data_path "${dataset_dir}/heldout_strict.pt" \
    --heldout_systems_data_path "${dataset_dir}/heldout_systems.pt" \
    --initialization_type "random" \
    --feature_type "consecutive_layer" \
    --eval_feature_type "consecutive_layer" \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --l2_penalty 1e-2 \
    --pn_ratio 1 \
    --eval_metrics "AUROC" "Spearman" \
    --train_sample_num -1 \
    --seeds 42 \
    --layer_index "attn_21" \
    --eval_layer_index "attn_21" \
    --projection_components -1 \
    --projection_type "pca" \
    --optimizer_type "adam" \
    --train_method "binary_only" \
    --best_single
