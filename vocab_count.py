import os
from tqdm import tqdm
import json
import torch
from multiprocessing import Pool, cpu_count
import numpy as np  # For efficient array operations

BATCH_SIZE = os.cpu_count() - 1


def get_text_list(folder_path):
    prompt_list = []

    # Open the file once and wrap the file object with `tqdm`
    with open(folder_path, "r") as f:
        for line in tqdm(f, desc="Reading JSONL", mininterval=1.0):
            data = json.loads(line)
            if "input" in data and "output" in data:
                prompt_list.append(data["input"])
            elif "text" in data:
                prompt_list.append(data["text"])

    return prompt_list


def count_freq(data_path, vocab_size, tokenizer, inherit_vocab_count):
    vocab_counts = [0 for _ in range(vocab_size)]
    # load data
    prompt_list = get_text_list(data_path)

    for i in tqdm(range(0, len(prompt_list), BATCH_SIZE), desc="Batch encoding"):
        batch_prompts = prompt_list[i : i + BATCH_SIZE]
        tokens_batch = tokenizer(batch_prompts)["input_ids"]

        # calculate prompt vocabs
        for tokens in tokens_batch:
            for token in tokens:
                vocab_counts[token] += 1

    # add inherit vocab if it's not none
    if inherit_vocab_count is not None:
        if os.path.exists(inherit_vocab_count):
            print(
                f"==> Load inherit_vocab_count and add it to current vocab_counts: path({inherit_vocab_count})"
            )
            inherit_vocab_count = torch.load(inherit_vocab_count)
            assert (
                len(inherit_vocab_count) == vocab_size
            ), f"inherit_vocab_count (size: {len(inherit_vocab_count)}) should have the same vocab size {vocab_size}"
            for token, i_count in enumerate(inherit_vocab_count):
                vocab_counts[token] += int(i_count)
        else:
            print(f"==> No valid inherit vocabulary count path, skip inheritance!")

    return vocab_counts


# def count_recursive(vocab_size, vocab_counts, old_bytes_list):
#     recursive_counts = [0 for _ in range(vocab_size)]

#     for i in tqdm(range(len(old_bytes_list))):
#         token_bytes = old_bytes_list[i]
#         t_count = vocab_counts[i]
#         b_len = len(token_bytes)
#         if t_count > 0 and b_len > 1:
#             for j in range(1, b_len):
#                 for k in range(b_len + 1 - j):
#                     sub_token = token_bytes[k : j + k]
#                     if sub_token in old_bytes_list:
#                         recursive_counts[old_bytes_list.index(sub_token)] += t_count

#     return recursive_counts


def count_recursive_worker(args):
    chunk_indices, vocab_counts, old_bytes_list, bytes_to_idx = args
    chunk_counts = [0] * len(old_bytes_list)

    for i in chunk_indices:
        t_count = vocab_counts[i]
        token_bytes = old_bytes_list[i]
        b_len = len(token_bytes)
        if t_count > 0 and b_len > 1:
            for j in range(1, b_len):
                for k in range(b_len + 1 - j):
                    sub_token = token_bytes[k:j + k]
                    if sub_token in bytes_to_idx:
                        chunk_counts[bytes_to_idx[sub_token]] += t_count

    return chunk_counts

def count_recursive(vocab_size, vocab_counts, old_bytes_list):
    # Ensure consistency
    if len(vocab_counts) != vocab_size:
        raise ValueError(f"vocab_counts length ({len(vocab_counts)}) must match vocab_size ({vocab_size})")
    if len(old_bytes_list) != vocab_size:
        raise ValueError(f"old_bytes_list length ({len(old_bytes_list)}) must match vocab_size ({vocab_size})")

    bytes_to_idx = {token: i for i, token in enumerate(old_bytes_list)}
    num_processes = max(1, cpu_count() - 1)
    indices = list(range(vocab_size))  # Use vocab_size directly
    chunk_size = max(1, len(indices) // num_processes)
    chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
    worker_args = [(chunk, vocab_counts, old_bytes_list, bytes_to_idx) for chunk in chunks]

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(count_recursive_worker, worker_args),
                           total=len(chunks),
                           desc="Processing chunks"))

    recursive_counts = np.sum(results, axis=0).tolist()
    return recursive_counts