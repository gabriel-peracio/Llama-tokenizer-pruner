import os
import torch
import json
import shutil
from tqdm import tqdm


def get_special_token_ids(tokenizer, model_path):
    """
    Automatically detect all special/added tokens that should be preserved.
    Returns a set of token IDs to always keep during pruning.
    """
    special_ids = set()

    # Method 1: added_tokens_decoder from tokenizer_config
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r") as f:
            config = json.load(f)
            if "added_tokens_decoder" in config:
                for token_id in config["added_tokens_decoder"].keys():
                    special_ids.add(int(token_id))

    # Method 2: Special tokens from tokenizer object
    if hasattr(tokenizer, "all_special_ids"):
        special_ids.update(tokenizer.all_special_ids)

    # Method 3: added_tokens from tokenizer.json
    tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, "r") as f:
            tok_json = json.load(f)
            if "added_tokens" in tok_json:
                for token in tok_json["added_tokens"]:
                    special_ids.add(token["id"])

    # Method 4: Detect byte fallback tokens (usually first 256 or 512 tokens in BPE)
    # These are critical for the tokenizer to function
    vocab = tokenizer.get_vocab()
    reverse_vocab = {v: k for k, v in vocab.items()}

    # Check if we have byte tokens (common pattern: <0x00> to <0xFF>)
    byte_token_pattern_count = 0
    for i in range(min(512, len(vocab))):  # Check first 512 tokens
        token = reverse_vocab.get(i, "")
        if token.startswith("<0x") and token.endswith(">"):
            special_ids.add(i)
            byte_token_pattern_count += 1

    print(f"Found {len(special_ids)} special/added tokens to preserve")
    if byte_token_pattern_count > 0:
        print(f"  - {byte_token_pattern_count} byte fallback tokens")

    return special_ids


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


def get_new_vocab_and_map(
    old_bytes_list, old_vocab_size, vocab_counts, recur_counts, special_token_ids
):
    """
    Create new vocabulary by keeping only tokens that:
    1. Are special tokens (always preserve)
    2. Have non-zero counts (appear in dataset or as sub-tokens)

    Args:
        old_bytes_list: List of token byte representations
        old_vocab_size: Original vocabulary size (from tokenizer)
        vocab_counts: Count of direct occurrences in dataset
        recur_counts: Count of occurrences as sub-tokens
        special_token_ids: Set of token IDs to always preserve
    """
    new_bytes_list = []
    mapping_new2old = []

    # Add regular tokens with non-zero counts OR special tokens
    for token_idx in range(len(old_bytes_list)):
        if (
            token_idx in special_token_ids
            or vocab_counts[token_idx] + recur_counts[token_idx] > 0
        ):
            new_bytes_list.append(old_bytes_list[token_idx])
            mapping_new2old.append(token_idx)

    # Handle any tokens beyond old_bytes_list (shouldn't happen with our fix, but be safe)
    extended_range_count = old_vocab_size - len(old_bytes_list)
    if extended_range_count > 0:
        print(f"Warning: {extended_range_count} tokens exist beyond old_bytes_list")
        for token_idx in range(len(old_bytes_list), old_vocab_size):
            if token_idx in special_token_ids:
                print(f"  Preserving special token at index {token_idx}")
                mapping_new2old.append(token_idx)

    print(f"Vocabulary size: {old_vocab_size} => {len(mapping_new2old)}")
    print(
        f"  - Special tokens preserved: {sum(1 for i in mapping_new2old if i in special_token_ids)}"
    )
    print(
        f"  - Used tokens preserved: {sum(1 for i in mapping_new2old if i not in special_token_ids)}"
    )

    return new_bytes_list, mapping_new2old


def save_vocab(bytes_list, token_mapping, output_path, old_model_path):
    # Create reverse mapping for quick lookups
    old_to_new = {old: new for new, old in enumerate(token_mapping)}

    # Copy and modify each file
    # 1. Copy files that don't need modification
    files_to_copy = [
        "special_tokens_map.json",
        "chat_template.jinja",
        "README.md",
    ]
    for filename in files_to_copy:
        src_path = os.path.join(old_model_path, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(output_path, filename))
            print(f"Copied {filename}")

    # 2. config.json
    with open(os.path.join(old_model_path, "config.json"), "r") as f:
        config = json.load(f)

    # Update vocab_size
    config["vocab_size"] = len(token_mapping)

    # Update token IDs if they exist and are in the mapping
    token_id_keys = ["bos_token_id", "eos_token_id", "pad_token_id"]
    for key in token_id_keys:
        if key in config and config[key] is not None and config[key] in old_to_new:
            old_val = config[key]
            config[key] = old_to_new[old_val]
            print(f"Updated config.{key}: {old_val} -> {config[key]}")

    # Update vision/image token indices if they exist (for multimodal models like Gemma)
    vision_keys = ["boi_token_index", "eoi_token_index", "image_token_index"]
    for key in vision_keys:
        if key in config and config[key] in old_to_new:
            old_val = config[key]
            config[key] = old_to_new[old_val]
            print(f"Updated config.{key}: {old_val} -> {config[key]}")

    # Update text_config.vocab_size if it exists (for models with text_config like Gemma)
    if "text_config" in config and isinstance(config["text_config"], dict):
        config["text_config"]["vocab_size"] = len(token_mapping)

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
            "pad_token_id",
            # "finetune_right_pad_id"
        ]:
            if key in gen_config and gen_config[key] is not None:
                old_id = gen_config[key]

                # Handle both single IDs and lists of IDs
                if isinstance(old_id, list):
                    # Update list of IDs
                    new_ids = []
                    for id_val in old_id:
                        if id_val in old_to_new:
                            new_ids.append(old_to_new[id_val])
                        else:
                            print(
                                f"WARNING: Could not find mapping for generation_config.{key}[{id_val}]"
                            )
                    if new_ids:
                        gen_config[key] = new_ids
                        print(f"Updated generation_config.{key}: {old_id} -> {new_ids}")
                else:
                    # Update single ID
                    if old_id in old_to_new:
                        gen_config[key] = old_to_new[old_id]
                        print(
                            f"Updated generation_config.{key}: {old_id} -> {old_to_new[old_id]}"
                        )
                    else:
                        print(
                            f"WARNING: Could not find mapping for generation_config.{key}: {old_id}"
                        )

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

    # Update post_processor special token IDs if they exist
    if "post_processor" in tokenizer and tokenizer["post_processor"] is not None:
        if "special_tokens" in tokenizer["post_processor"]:
            for token_name, token_data in tokenizer["post_processor"][
                "special_tokens"
            ].items():
                if "ids" in token_data and len(token_data["ids"]) > 0:
                    old_id = token_data["ids"][0]
                    if old_id in old_to_new:
                        token_data["ids"][0] = old_to_new[old_id]
                        print(
                            f"Updated post_processor special token {token_name}: {old_id} -> {old_to_new[old_id]}"
                        )

    # Update vocabulary
    vocabulary = {}
    for i, token_bytes in enumerate(bytes_list):
        token = token_bytes.decode("utf-8")
        vocabulary[token] = i

    # Update merges (keeping only valid ones AND their components)
    original_merges = tokenizer["model"]["merges"]
    valid_merges = []
    for merge in original_merges:
        # Handle both string format ("token1 token2") and tuple format (("token1", "token2"))
        if isinstance(merge, str):
            parts = merge.split()
            if len(parts) != 2:
                continue  # Skip invalid merge entries
            part1, part2 = parts
        else:
            # Already a tuple/list
            part1, part2 = merge

        merged = part1 + part2
        # Only keep merge if both the parts AND the result exist in vocabulary
        if part1 in vocabulary and part2 in vocabulary and merged in vocabulary:
            valid_merges.append(merge)

    tokenizer["model"]["vocab"] = vocabulary
    tokenizer["model"]["merges"] = valid_merges

    with open(os.path.join(output_path, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f, indent=2, ensure_ascii=False)

    # 6. Create standalone vocab.json (some tokenizers need this)
    with open(os.path.join(output_path, "vocab.json"), "w") as f:
        json.dump(vocabulary, f, indent=2, ensure_ascii=False)
    print(f"Saved vocab.json with {len(vocabulary)} tokens")

    # 7. Create standalone merges.txt (some tokenizers need this)
    with open(os.path.join(output_path, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for merge in valid_merges:
            if isinstance(merge, str):
                # Already in string format
                f.write(f"{merge}\n")
            else:
                # Tuple/list format
                part1, part2 = merge
                f.write(f"{part1} {part2}\n")
    print(f"Saved merges.txt with {len(valid_merges)} merges")

    # 8. Create added_tokens.json from tokenizer.json added_tokens
    added_tokens_dict = {}
    if "added_tokens" in tokenizer:
        for token_info in tokenizer["added_tokens"]:
            added_tokens_dict[token_info["content"]] = token_info["id"]
    with open(os.path.join(output_path, "added_tokens.json"), "w") as f:
        json.dump(added_tokens_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved added_tokens.json with {len(added_tokens_dict)} tokens")

    # Save mapping index
    token_mapping_path = os.path.join(output_path, "token_mapping.torch")
    torch.save(torch.LongTensor(token_mapping), token_mapping_path)
