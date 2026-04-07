#!/bin/sh

set -e

config="examples/nemotron-nano.json"
model_path="/root/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
output_path="/root/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16-Pruned"
dataset="/root/datasets/rawDataset*.jsonl.zst"
inherit_vocab_count="" # optional

set -x

if [ -n "$inherit_vocab_count" ]; then
    tokenizer-pruner \
        --config "$config" \
        --model_path "$model_path" \
        --output_path "$output_path" \
        --dataset "$dataset" \
        --inherit_vocab_count "$inherit_vocab_count"
else
    tokenizer-pruner \
        --config "$config" \
        --model_path "$model_path" \
        --output_path "$output_path" \
        --dataset "$dataset"
fi

