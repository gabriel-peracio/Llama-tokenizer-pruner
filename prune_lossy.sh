#!/bin/sh


old_model_path="/mnt/f/LLM/Meta-Llama-3.1-8B-bnb-4bit"
new_model_path="/mnt/f/LLM/Meta-Llama-3.1-8B-bnb-4bit-lossy-pruned-vocab"
support_data="./combined.json"
inherit_vocab_count="" # optional
# target_size=20000
target_size=31745

# run pruning, check whether optional args are exists
if [ -z "$inherit_vocab_count" ]; then
    cmd="python main.py --old_model_path $old_model_path --new_model_path $new_model_path --support_data $support_data --target_size $target_size"
else
    cmd="python main.py --old_model_path $old_model_path --new_model_path $new_model_path --support_data $support_data --inherit_vocab_count $inherit_vocab_count --target_size $target_size"
fi
echo $cmd
$cmd