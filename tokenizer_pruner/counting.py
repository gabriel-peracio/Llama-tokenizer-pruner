"""Token frequency counting with multiprocessing support."""

import os
import json
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from tqdm import tqdm

BATCH_SIZE = max(1, os.cpu_count() - 1)


def get_text_list(folder_path: str) -> list[str]:
    """Load text data from a JSONL file.

    Supports two formats:
    - {"input": ..., "output": ...} - uses "input" field
    - {"text": ...} - uses "text" field
    """
    prompt_list = []

    with open(folder_path, "r") as f:
        for line in tqdm(f, desc="Reading JSONL", mininterval=1.0):
            data = json.loads(line)
            if "input" in data and "output" in data:
                prompt_list.append(data["input"])
            elif "text" in data:
                prompt_list.append(data["text"])

    return prompt_list


def count_freq(
    data_path: str,
    vocab_size: int,
    tokenizer,
    inherit_vocab_count: str | None = None,
) -> list[int]:
    """Count token frequencies in a dataset.

    Args:
        data_path: Path to JSONL file with text data
        vocab_size: Size of the vocabulary
        tokenizer: HuggingFace tokenizer
        inherit_vocab_count: Optional path to previous vocab counts to merge

    Returns:
        List of token counts indexed by token ID
    """
    vocab_counts = [0 for _ in range(vocab_size)]
    prompt_list = get_text_list(data_path)

    out_of_range_count = 0
    for i in tqdm(range(0, len(prompt_list), BATCH_SIZE), desc="Batch encoding"):
        batch_prompts = prompt_list[i : i + BATCH_SIZE]

        # Try batch encoding first, fall back to individual encoding for custom tokenizers
        try:
            result = tokenizer(batch_prompts)
            if result is not None and "input_ids" in result:
                tokens_batch = result["input_ids"]
            else:
                raise ValueError("Batch encoding returned invalid result")
        except (ValueError, TypeError, AttributeError):
            # Fall back to individual encoding
            tokens_batch = [tokenizer.encode(text) for text in batch_prompts]

        for tokens in tokens_batch:
            for token in tokens:
                if token < vocab_size:
                    vocab_counts[token] += 1
                else:
                    out_of_range_count += 1

    if out_of_range_count > 0:
        print(
            f"Warning: {out_of_range_count} token occurrences had IDs >= vocab_size ({vocab_size}), skipped"
        )

    # Add inherited vocab counts if provided
    if inherit_vocab_count is not None:
        if os.path.exists(inherit_vocab_count):
            print(
                f"==> Load inherit_vocab_count and add it to current vocab_counts: path({inherit_vocab_count})"
            )
            inherited = torch.load(inherit_vocab_count, weights_only=False)
            if len(inherited) != vocab_size:
                raise ValueError(
                    f"inherit_vocab_count (size: {len(inherited)}) should have the same vocab size {vocab_size}"
                )
            for token, i_count in enumerate(inherited):
                vocab_counts[token] += int(i_count)
        else:
            print(f"==> No valid inherit vocabulary count path, skip inheritance!")

    return vocab_counts


def _count_recursive_worker(args):
    """Worker function for parallel recursive counting."""
    chunk_indices, vocab_counts, old_bytes_list, bytes_to_idx = args
    chunk_counts = [0] * len(old_bytes_list)

    for i in chunk_indices:
        t_count = vocab_counts[i]
        token_bytes = old_bytes_list[i]
        b_len = len(token_bytes)
        if t_count > 0 and b_len > 1:
            for j in range(1, b_len):
                for k in range(b_len + 1 - j):
                    sub_token = token_bytes[k : j + k]
                    if sub_token in bytes_to_idx:
                        chunk_counts[bytes_to_idx[sub_token]] += t_count

    return chunk_counts


def count_recursive(
    vocab_size: int,
    vocab_counts: list[int],
    old_bytes_list: list[bytes],
) -> list[int]:
    """Count recursive sub-token occurrences using multiprocessing.

    For each token that appears in the dataset, count how many times
    its sub-tokens would be needed if the token were removed.

    Args:
        vocab_size: Size of vocabulary
        vocab_counts: Direct token counts from dataset
        old_bytes_list: List of token byte representations

    Returns:
        List of recursive counts indexed by token ID
    """
    if len(vocab_counts) != vocab_size:
        raise ValueError(
            f"vocab_counts length ({len(vocab_counts)}) must match vocab_size ({vocab_size})"
        )
    if len(old_bytes_list) != vocab_size:
        raise ValueError(
            f"old_bytes_list length ({len(old_bytes_list)}) must match vocab_size ({vocab_size})"
        )

    bytes_to_idx = {token: i for i, token in enumerate(old_bytes_list)}
    num_processes = max(1, cpu_count() - 1)
    indices = list(range(vocab_size))
    chunk_size = max(1, len(indices) // num_processes)
    chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]
    worker_args = [
        (chunk, vocab_counts, old_bytes_list, bytes_to_idx) for chunk in chunks
    ]

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(_count_recursive_worker, worker_args),
                total=len(chunks),
                desc="Processing chunks",
            )
        )

    recursive_counts = np.sum(results, axis=0).tolist()
    return recursive_counts
