#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

old_model_path="/mnt/f/LLM/Meta-Llama-3.1-8B-bnb-4bit"
new_model_path="/mnt/f/LLM/Meta-Llama-3.1-8B-bnb-4bit-pruned-vocab"
support_data="./dataset.json"
inherit_vocab_count="" # optional

# run pruning, check whether optional args are exists
if [ -z "$inherit_vocab_count" ]; then
    cmd="python main.py --old_model_path $old_model_path --new_model_path $new_model_path --support_data $support_data"
else
    cmd="python main.py --old_model_path $old_model_path --new_model_path $new_model_path --support_data $support_data --inherit_vocab_count $inherit_vocab_count"
fi
echo $cmd
$cmd

# run check the new tokenizer works as the same as old tokenizer in support data
cmd="python check.py --old_model_path=$old_model_path --new_model_path=$new_model_path --support_data=$support_data"
echo $cmd
$cmd
