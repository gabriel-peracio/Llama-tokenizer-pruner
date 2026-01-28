#!/bin/sh


# model_path="/mnt/f/LLM/Meta-Llama-3.1-8B-bnb-4bit"
# output_path="/mnt/f/LLM/Meta-Llama-3.1-8B-bnb-4bit-lossy-pruned-vocab"
old_model_path="/mnt/e/models/unsloth/gemma-3-27b-pt-unsloth-bnb-4bit"
new_model_path="/mnt/e/models/unsloth/gemma-3-27b-pt-unsloth-bnb-4bit-pruned-vocab"
dataset="./data.jsonl"
inherit_vocab_count="" # optional
# target_vocab_size=20000
target_vocab_size=32768 # 58888 will make 65536 vocab (with all the special tokens) for gemma-3-27b

# run pruning, check whether optional args are exists
if [ -z "$inherit_vocab_count" ]; then
    cmd="python main.py --model_path $model_path --output_path $output_path --dataset $dataset --target_vocab_size $target_vocab_size"
else
    cmd="python main.py --model_path $model_path --output_path $output_path --dataset $dataset --inherit_vocab_count $inherit_vocab_count --target_vocab_size $target_vocab_size"
fi
echo $cmd
$cmd