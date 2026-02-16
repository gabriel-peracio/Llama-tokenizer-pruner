"""Token frequency counting with multiprocessing support."""

import glob
import io
import os
import json
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import zstandard as zstd
from tqdm import tqdm

BATCH_SIZE = max(1, os.cpu_count() - 1)


def resolve_dataset_paths(pattern: str) -> list[str]:
    """Resolve a dataset argument into a sorted list of file paths.

    Accepts either a single file path or a glob pattern.
    Raises FileNotFoundError if no files match.
    """
    if os.path.isfile(pattern):
        return [pattern]

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files matched the dataset pattern: {pattern}"
        )
    return paths


def get_text_list(file_path: str) -> list[str]:
    """Load text data from a JSONL or JSONL.zst file.

    Supports two JSONL row formats:
    - {"input": ..., "output": ...} - uses "input" field
    - {"text": ...} - uses "text" field

    The file may be plain text (.jsonl) or zstandard-compressed (.jsonl.zst).
    """
    prompt_list = []

    if file_path.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        with open(file_path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in tqdm(text_stream, desc=f"Reading {os.path.basename(file_path)}", mininterval=1.0):
                    data = json.loads(line)
                    if "input" in data and "output" in data:
                        prompt_list.append(data["input"])
                    elif "text" in data:
                        prompt_list.append(data["text"])
    else:
        with open(file_path, "r") as f:
            for line in tqdm(f, desc=f"Reading {os.path.basename(file_path)}", mininterval=1.0):
                data = json.loads(line)
                if "input" in data and "output" in data:
                    prompt_list.append(data["input"])
                elif "text" in data:
                    prompt_list.append(data["text"])

    return prompt_list


def _count_file(
    file_path: str,
    vocab_size: int,
    tokenizer,
    vocab_counts: list[int],
) -> int:
    """Count token frequencies from a single file, accumulating into vocab_counts.

    Returns the number of out-of-range token occurrences encountered.
    """
    prompt_list = get_text_list(file_path)

    out_of_range_count = 0
    for i in tqdm(range(0, len(prompt_list), BATCH_SIZE), desc=f"Encoding {os.path.basename(file_path)}"):
        batch_prompts = prompt_list[i : i + BATCH_SIZE]

        try:
            result = tokenizer(batch_prompts)
            if result is not None and "input_ids" in result:
                tokens_batch = result["input_ids"]
            else:
                raise ValueError("Batch encoding returned invalid result")
        except (ValueError, TypeError, AttributeError):
            tokens_batch = [tokenizer.encode(text) for text in batch_prompts]

        for tokens in tokens_batch:
            for token in tokens:
                if token < vocab_size:
                    vocab_counts[token] += 1
                else:
                    out_of_range_count += 1

    return out_of_range_count


def count_freq(
    data_path: str | list[str],
    vocab_size: int,
    tokenizer,
    inherit_vocab_count: str | None = None,
) -> list[int]:
    """Count token frequencies in a dataset.

    Args:
        data_path: A single file path, a glob pattern, or an already-resolved
                   list of file paths.  Files are processed one at a time so
                   that multi-file globs do not exhaust memory.
        vocab_size: Size of the vocabulary
        tokenizer: HuggingFace tokenizer
        inherit_vocab_count: Optional path to previous vocab counts to merge

    Returns:
        List of token counts indexed by token ID
    """
    if isinstance(data_path, str):
        file_paths = resolve_dataset_paths(data_path)
    else:
        file_paths = data_path

    vocab_counts = [0 for _ in range(vocab_size)]

    out_of_range_count = 0
    for file_path in tqdm(file_paths, desc="Dataset files", disable=len(file_paths) == 1):
        out_of_range_count += _count_file(file_path, vocab_size, tokenizer, vocab_counts)

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
