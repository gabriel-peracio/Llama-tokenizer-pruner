#!/bin/sh

set -e

export CUDA_VISIBLE_DEVICES=1

config="examples/qwen3.json"
model_path="/mnt/nvme/share/Models/LLM/Qwen/Qwen3-0.6B-Base"
output_path="/mnt/nvme/share/Models/LLM/Qwen/Qwen3-0.6B-Base-pruned-vocab-alt-tokenizer"
dataset="/mnt/nvme/share/Datasets/filler-alttokenizer-32k-0.jsonl"
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
