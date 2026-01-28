"""Vocabulary operations: special token detection, pruning, and saving."""

import json
import os
import re
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from .config import ModelConfig


# GPT-2 style byte-to-unicode mapping (used by Llama 3, GPT-2, etc.)
# Maps bytes 0-255 to printable unicode characters
def _build_gpt2_byte_mapping() -> dict[int, str]:
    """Build the GPT-2 byte-to-unicode character mapping.

    This maps bytes 0-255 to visible Unicode characters:
    - Printable ASCII (33-126) maps to itself
    - Other bytes map to Unicode chars starting at U+0100
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


GPT2_BYTE_TO_CHAR = _build_gpt2_byte_mapping()
GPT2_CHAR_TO_BYTE = {v: k for k, v in GPT2_BYTE_TO_CHAR.items()}


def _update_post_processor_ids(post_processor: dict, old_to_new: dict) -> None:
    """Recursively update special token IDs in post_processor.

    Handles both flat and nested (Sequence) post_processor structures.
    """
    if post_processor is None:
        return

    # Handle special_tokens at this level
    if "special_tokens" in post_processor:
        for token_name, token_data in post_processor["special_tokens"].items():
            if "ids" in token_data and len(token_data["ids"]) > 0:
                old_id = token_data["ids"][0]
                if old_id in old_to_new:
                    token_data["ids"][0] = old_to_new[old_id]
                    print(
                        f"Updated post_processor special token {token_name}: {old_id} -> {old_to_new[old_id]}"
                    )

    # Handle nested processors (Sequence type)
    if "processors" in post_processor:
        for processor in post_processor["processors"]:
            _update_post_processor_ids(processor, old_to_new)


def _is_gpt2_byte_token(token: str) -> bool:
    """Check if a token is a single-byte token in GPT-2 encoding."""
    if len(token) != 1:
        return False
    return token in GPT2_CHAR_TO_BYTE


def get_special_token_ids(
    tokenizer,
    model_path: str,
    config: ModelConfig | None = None,
    max_token_id: int | None = None,
) -> set[int]:
    """Detect all special/added tokens that should be preserved.

    Uses a multi-layer detection approach:
    1. added_tokens_decoder from tokenizer_config.json
    2. Special tokens from tokenizer object (all_special_ids)
    3. added_tokens from tokenizer.json
    4. Byte fallback tokens - both <0x00> style AND GPT-2 style encoding
    5. Preserve patterns from config

    Args:
        tokenizer: HuggingFace tokenizer
        model_path: Path to model directory
        config: Optional ModelConfig with preserve patterns
        max_token_id: Maximum token ID to consider (for vocab size limits)

    Returns:
        Set of token IDs to always preserve
    """
    special_ids = set()

    # Method 1: added_tokens_decoder from tokenizer_config
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r") as f:
            tok_config = json.load(f)
            if "added_tokens_decoder" in tok_config:
                for token_id in tok_config["added_tokens_decoder"].keys():
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

    # Method 4: Detect byte fallback tokens
    vocab = tokenizer.get_vocab()
    reverse_vocab = {v: k for k, v in vocab.items()}

    # Get byte fallback pattern from config or use default
    if config is not None:
        byte_pattern = config.get_byte_fallback_pattern()
    else:
        byte_pattern = re.compile(r"<0x[0-9A-Fa-f]{2}>")

    byte_token_count = 0
    gpt2_byte_count = 0

    # Check first 512 tokens for byte patterns
    for i in range(min(512, len(vocab))):
        token = reverse_vocab.get(i, "")

        # Check for <0x..> style byte tokens (Qwen, etc.)
        if byte_pattern.match(token):
            special_ids.add(i)
            byte_token_count += 1
        # Check for GPT-2 style single-char byte tokens (Llama 3, GPT-2, etc.)
        elif _is_gpt2_byte_token(token):
            special_ids.add(i)
            gpt2_byte_count += 1

    # Method 5: Match preserve_tokens patterns from config
    if config is not None:
        preserve_patterns = config.get_preserve_patterns()
        if preserve_patterns:
            pattern_match_count = 0
            for token, token_id in vocab.items():
                for pattern in preserve_patterns:
                    if pattern.match(token):
                        special_ids.add(token_id)
                        pattern_match_count += 1
                        break

            if pattern_match_count > 0:
                print(f"  - {pattern_match_count} tokens matched preserve patterns")

    # Filter out token IDs beyond max_token_id if specified
    if max_token_id is not None:
        excluded = {tid for tid in special_ids if tid >= max_token_id}
        if excluded:
            print(
                f"  - Excluding {len(excluded)} special tokens beyond vocab limit (id >= {max_token_id})"
            )
        special_ids = {tid for tid in special_ids if tid < max_token_id}

    print(f"Found {len(special_ids)} special/added tokens to preserve")
    if byte_token_count > 0:
        print(f"  - {byte_token_count} byte fallback tokens (<0x..> style)")
    if gpt2_byte_count > 0:
        print(f"  - {gpt2_byte_count} byte fallback tokens (GPT-2 style)")

    return special_ids


def reduce_to_target_size(
    old_vocab_size: int,
    target_vocab_size: int,
    vocab_counts: list[int],
    recur_counts: list[int],
    old_bytes_list: list[bytes],
) -> tuple[list[int], list[int]]:
    """Reduce vocabulary to target size by removing least-used tokens.

    Tokens are removed only if they can be represented by existing sub-tokens.

    Args:
        old_vocab_size: Original vocabulary size
        target_vocab_size: Desired vocabulary size
        vocab_counts: Direct token counts
        recur_counts: Recursive sub-token counts
        old_bytes_list: List of token byte representations

    Returns:
        Tuple of (modified vocab_counts, modified recur_counts)
    """
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
            # Check if token can be represented by sub-tokens
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
    old_bytes_list: list[bytes],
    old_vocab_size: int,
    vocab_counts: list[int],
    recur_counts: list[int],
    special_token_ids: set[int],
) -> tuple[list[bytes], list[int]]:
    """Create new vocabulary by keeping only used and special tokens.

    Args:
        old_bytes_list: List of token byte representations
        old_vocab_size: Original vocabulary size
        vocab_counts: Direct token counts
        recur_counts: Recursive sub-token counts
        special_token_ids: Set of token IDs to always preserve

    Returns:
        Tuple of (new_bytes_list, mapping_new2old)
    """
    new_bytes_list = []
    mapping_new2old = []

    # Add tokens with non-zero counts OR special tokens
    for token_idx in range(len(old_bytes_list)):
        if (
            token_idx in special_token_ids
            or vocab_counts[token_idx] + recur_counts[token_idx] > 0
        ):
            new_bytes_list.append(old_bytes_list[token_idx])
            mapping_new2old.append(token_idx)

    # Handle any tokens beyond old_bytes_list (safety check)
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


def save_vocab(
    bytes_list: list[bytes],
    token_mapping: list[int],
    output_path: str,
    old_model_path: str,
) -> None:
    """Save pruned vocabulary and update all tokenizer files.

    Args:
        bytes_list: New vocabulary as byte representations
        token_mapping: Mapping from new to old token indices
        output_path: Output directory
        old_model_path: Original model directory
    """
    old_to_new = {old: new for new, old in enumerate(token_mapping)}

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

    config["vocab_size"] = len(token_mapping)

    # Update token IDs
    token_id_keys = ["bos_token_id", "eos_token_id", "pad_token_id"]
    for key in token_id_keys:
        if key in config and config[key] is not None and config[key] in old_to_new:
            old_val = config[key]
            config[key] = old_to_new[old_val]
            print(f"Updated config.{key}: {old_val} -> {config[key]}")

    # Update vision/image token indices for multimodal models
    vision_keys = ["boi_token_index", "eoi_token_index", "image_token_index"]
    for key in vision_keys:
        if key in config and config[key] in old_to_new:
            old_val = config[key]
            config[key] = old_to_new[old_val]
            print(f"Updated config.{key}: {old_val} -> {config[key]}")

    # Update text_config.vocab_size for models with nested config
    if "text_config" in config and isinstance(config["text_config"], dict):
        config["text_config"]["vocab_size"] = len(token_mapping)

    with open(os.path.join(output_path, "config.json"), "w+") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 3. generation_config.json
    gen_config_path = os.path.join(old_model_path, "generation_config.json")
    if os.path.exists(gen_config_path):
        with open(gen_config_path, "r") as f:
            gen_config = json.load(f)

        for key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            if key in gen_config and gen_config[key] is not None:
                old_id = gen_config[key]

                if isinstance(old_id, list):
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
                    if old_id in old_to_new:
                        gen_config[key] = old_to_new[old_id]
                        print(
                            f"Updated generation_config.{key}: {old_id} -> {old_to_new[old_id]}"
                        )
                    else:
                        print(
                            f"WARNING: Could not find mapping for generation_config.{key}: {old_id}"
                        )

        output_gen_config = os.path.join(output_path, "generation_config.json")
        with open(output_gen_config, "w+") as f:
            json.dump(gen_config, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
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

    # Update post_processor special token IDs (handles nested Sequence processors)
    if "post_processor" in tokenizer and tokenizer["post_processor"] is not None:
        _update_post_processor_ids(tokenizer["post_processor"], old_to_new)

    # Update vocabulary
    vocabulary = {}
    for i, token_bytes in enumerate(bytes_list):
        token = token_bytes.decode("utf-8")
        vocabulary[token] = i

    # Update merges
    original_merges = tokenizer["model"]["merges"]
    valid_merges = []
    for merge in original_merges:
        if isinstance(merge, str):
            parts = merge.split()
            if len(parts) != 2:
                continue
            part1, part2 = parts
        else:
            part1, part2 = merge

        merged = part1 + part2
        if part1 in vocabulary and part2 in vocabulary and merged in vocabulary:
            valid_merges.append(merge)

    tokenizer["model"]["vocab"] = vocabulary
    tokenizer["model"]["merges"] = valid_merges

    with open(os.path.join(output_path, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f, indent=2, ensure_ascii=False)

    # 6. vocab.json
    with open(os.path.join(output_path, "vocab.json"), "w") as f:
        json.dump(vocabulary, f, indent=2, ensure_ascii=False)
    print(f"Saved vocab.json with {len(vocabulary)} tokens")

    # 7. merges.txt
    with open(os.path.join(output_path, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for merge in valid_merges:
            if isinstance(merge, str):
                f.write(f"{merge}\n")
            else:
                part1, part2 = merge
                f.write(f"{part1} {part2}\n")
    print(f"Saved merges.txt with {len(valid_merges)} merges")

    # 8. added_tokens.json
    added_tokens_dict = {}
    if "added_tokens" in tokenizer:
        for token_info in tokenizer["added_tokens"]:
            added_tokens_dict[token_info["content"]] = token_info["id"]
    with open(os.path.join(output_path, "added_tokens.json"), "w") as f:
        json.dump(added_tokens_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved added_tokens.json with {len(added_tokens_dict)} tokens")

    # 9. token_mapping.torch
    token_mapping_path = os.path.join(output_path, "token_mapping.torch")
    torch.save(torch.LongTensor(token_mapping), token_mapping_path)
