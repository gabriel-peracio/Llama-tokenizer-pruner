import os
import torch
import json
import shutil
from tqdm import tqdm


def reduce_to_target_size(
    old_vocab_size, target_vocab_size, vocab_counts, recur_counts, old_bytes_list
):
    total_count_with_idx = [
        (vocab_counts[i] + recur_counts[i], i) for i in range(old_vocab_size)
    ]
    sorted_count_with_idx = sorted(total_count_with_idx, key=lambda x: x[0])
    remove_count = 0
    remove_target = old_vocab_size - target_vocab_size

    for i in tqdm(range(len(sorted_count_with_idx))):
        token_count, token_idx = sorted_count_with_idx[i]
        if remove_count >= remove_target:
            continue
        elif token_count == 0:
            remove_count += 1
        elif len(old_bytes_list[token_idx]) > 1:
            # whether it can be represented by sub-token
            token = old_bytes_list[token_idx]
            b_len = len(token)
            for j in range(1, b_len):
                if (token[:j] in old_bytes_list) and (token[j:] in old_bytes_list):
                    parta_index = old_bytes_list.index(token[:j])
                    partb_index = old_bytes_list.index(token[j:])
                    if (vocab_counts[parta_index] + recur_counts[parta_index] > 0) and (
                        vocab_counts[partb_index] + recur_counts[partb_index] > 0
                    ):
                        vocab_counts[token_idx] = 0
                        recur_counts[token_idx] = 0
                        remove_count += 1
                        break

    if remove_count < remove_target:
        print(
            f"Failed to reach the target size, could only remove {remove_count} tokens"
        )
    return vocab_counts, recur_counts


def get_new_vocab_and_map(old_bytes_list, old_vocab_size, vocab_counts, recur_counts):
    new_bytes_list = []
    mapping_new2old = []

    # fmt:off
    special_tokens = [
        128000, 128001, 128004,  # bos, eos, pad
        128006, 128007, 128008, 128009, 128010,  # start_header_id, end_header_id, eom_id, eot_id, python_tag
        128002, 128003, 128005, # rst_0, rst_1, rst_2
        *range(128011, 128256) # rst_3 to rst_247
    ]
    # fmt:on

    # Add regular tokens with non-zero counts
    for token_idx in range(len(old_bytes_list)):
        if (
            token_idx in special_tokens
            or vocab_counts[token_idx] + recur_counts[token_idx] > 0
        ):
            new_bytes_list.append(old_bytes_list[token_idx])
            mapping_new2old.append(token_idx)

    # Add special tokens from extended range
    extended_range_count = old_vocab_size - len(old_bytes_list)
    print(f"Adding special tokens from extended range (count: {extended_range_count})")
    for token_idx in range(len(old_bytes_list), old_vocab_size):
        if token_idx in special_tokens:
            mapping_new2old.append(token_idx)

    print(f"Vocabulary size: {old_vocab_size} => {len(mapping_new2old)}")
    return new_bytes_list, mapping_new2old


def save_vocab(bytes_list, token_mapping, output_path, old_model_path):
    # Create reverse mapping for quick lookups
    old_to_new = {old: new for new, old in enumerate(token_mapping)}

    # Debug print
    print("Special token mapping:")
    for old_id in [128000, 128001, 128004]:
        print(f"  {old_id} -> {old_to_new.get(old_id, 'NOT FOUND')}")

    # Copy and modify each file
    # 1. special_tokens_map.json - straight copy
    shutil.copy2(
        os.path.join(old_model_path, "special_tokens_map.json"),
        os.path.join(output_path, "special_tokens_map.json"),
    )

    # 2. config.json
    with open(os.path.join(old_model_path, "config.json"), "r") as f:
        config = json.load(f)
    config.update(
        {
            "bos_token_id": old_to_new[128000],
            "eos_token_id": old_to_new[128001],
            "vocab_size": len(token_mapping),
        }
    )
    with open(os.path.join(output_path, "config.json"), "w+") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 3. generation_config.json
    gen_config_path = os.path.join(old_model_path, "generation_config.json")
    if os.path.exists(gen_config_path):
        with open(gen_config_path, "r") as f:
            gen_config = json.load(f)

        # Update the config
        for key in [
            "bos_token_id",
            "eos_token_id",
            # "finetune_right_pad_id"
        ]:
            old_id = gen_config[key]
            if old_id in old_to_new:
                gen_config[key] = old_to_new[old_id]
                print(f"Updated {key}: {old_id} -> {old_to_new[old_id]}")
            else:
                print(f"WARNING: Could not find mapping for {key}: {old_id}")

        # Save the updated config
        output_gen_config = os.path.join(output_path, "generation_config.json")
        with open(output_gen_config, "w+") as f:
            json.dump(gen_config, f, indent=2, ensure_ascii=False)
            f.flush()  # Force write to disk
            os.fsync(f.fileno())  # Make sure it's written
        print(f"Saved updated generation config to {output_gen_config}")

    # 4. tokenizer_config.json
    with open(os.path.join(old_model_path, "tokenizer_config.json"), "r") as f:
        tok_config = json.load(f)

    new_decoder = {}
    for old_id, token_info in tok_config["added_tokens_decoder"].items():
        old_id = int(old_id)
        if old_id in old_to_new:
            new_decoder[str(old_to_new[old_id])] = token_info
    tok_config["added_tokens_decoder"] = new_decoder

    with open(os.path.join(output_path, "tokenizer_config.json"), "w+") as f:
        json.dump(tok_config, f, indent=2, ensure_ascii=False)

    # 5. tokenizer.json
    with open(os.path.join(old_model_path, "tokenizer.json"), "r") as f:
        tokenizer = json.load(f)

    # Update added_tokens
    new_added_tokens = []
    for token in tokenizer["added_tokens"]:
        old_id = token["id"]
        if old_id in old_to_new:
            token["id"] = old_to_new[old_id]
            new_added_tokens.append(token)
    tokenizer["added_tokens"] = new_added_tokens

    # Update post_processor special token IDs
    tokenizer["post_processor"]["processors"][1]["special_tokens"]["<|begin_of_text|>"][
        "ids"
    ][0] = old_to_new[128000]

    # Update vocabulary
    vocabulary = {}
    for i, token_bytes in enumerate(bytes_list):
        token = token_bytes.decode("utf-8")
        vocabulary[token] = i

    # Update merges (keeping only valid ones AND their components)
    original_merges = tokenizer["model"]["merges"]
    valid_merges = []
    for merge in original_merges:
        parts = merge.split()
        # Only keep merge if both the parts AND the result exist in vocabulary
        merged = "".join(parts)  # This might need adjustment for the space char
        if all(part in vocabulary for part in parts) and merged in vocabulary:
            valid_merges.append(merge)

    tokenizer["model"]["vocab"] = vocabulary
    tokenizer["model"]["merges"] = valid_merges

    with open(os.path.join(output_path, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f, indent=2, ensure_ascii=False)

    # Save mapping index
    token_mapping_path = os.path.join(output_path, "token_mapping.torch")
    torch.save(torch.LongTensor(token_mapping), token_mapping_path)
