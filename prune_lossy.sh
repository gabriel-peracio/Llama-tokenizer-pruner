#!/bin/sh

set -e

config="examples/gemma3.json"
model_path="/mnt/e/models/unsloth/gemma-3-27b-pt-unsloth-bnb-4bit"
output_path="/mnt/e/models/unsloth/gemma-3-27b-pt-unsloth-bnb-4bit-pruned-vocab"
dataset="./data.jsonl"
inherit_vocab_count="" # optional
target_vocab_size=32768

set -x

if [ -n "$inherit_vocab_count" ]; then
    tokenizer-pruner \
        --config "$config" \
        --model_path "$model_path" \
        --output_path "$output_path" \
        --dataset "$dataset" \
        --target_vocab_size "$target_vocab_size" \
        --inherit_vocab_count "$inherit_vocab_count"
else
    tokenizer-pruner \
        --config "$config" \
        --model_path "$model_path" \
        --output_path "$output_path" \
        --dataset "$dataset" \
        --target_vocab_size "$target_vocab_size"
fi
