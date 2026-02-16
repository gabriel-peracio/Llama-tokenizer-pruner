#!/bin/sh

set -e

config="examples/gemma3.json"
model_path="/mnt/e/models/unsloth/gemma-3-27b-pt-unsloth-bnb-4bit"
output_path="/mnt/e/models/unsloth/gemma-3-27b-pt-unsloth-bnb-4bit-pruned-vocab"
dataset="./data.jsonl"
inherit_vocab_count="" # optional
target_vocab_size=32768

cmd="tokenizer-pruner --config $config --model_path $model_path --output_path $output_path --dataset $dataset --target_vocab_size $target_vocab_size"

if [ -n "$inherit_vocab_count" ]; then
    cmd="$cmd --inherit_vocab_count $inherit_vocab_count"
fi

echo "$cmd"
$cmd
