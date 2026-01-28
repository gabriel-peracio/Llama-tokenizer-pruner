#!/bin/sh

set -e

export CUDA_VISIBLE_DEVICES=1

# old_model_path="/mnt/f/LLM/Meta-Llama-3.1-8B-bnb-4bit"
# new_model_path="/mnt/f/LLM/Meta-Llama-3.1-8B-bnb-4bit-pruned-vocab"
model_path="/mnt/nvme/share/Models/LLM/Qwen/Qwen3-0.6B-Base"
output_path="/mnt/nvme/share/Models/LLM/Qwen/Qwen3-0.6B-Base-pruned-vocab-alt-tokenizer"
dataset="/mnt/nvme/share/Datasets/filler-alttokenizer-32k-0.jsonl"
# dataset="/mnt/nvme/share/Datasets/filler-32k-0.jsonl"
inherit_vocab_count="" # optional

# run pruning, check whether optional args are exists
if [ -z "$inherit_vocab_count" ]; then
    cmd="python main.py --model_path $model_path --output_path $output_path --dataset $dataset"
else
    cmd="python main.py --model_path $model_path --output_path $output_path --dataset $dataset --inherit_vocab_count $inherit_vocab_count"
fi
echo $cmd
$cmd

# run check the new tokenizer works as the same as old tokenizer in support data
cmd="python check.py --model_path=$model_path --output_path=$output_path --dataset=$dataset"
echo $cmd
$cmd
