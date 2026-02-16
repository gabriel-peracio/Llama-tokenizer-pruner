"""Command-line interface for tokenizer pruning."""

import argparse
import datetime
import os

import torch
from termcolor import colored
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2 import tokenization_gpt2

from .vocab_ops import GPT2_BYTE_TO_CHAR

# Monkey-patch for models that expect bytes_to_unicode (removed in newer transformers)
if not hasattr(tokenization_gpt2, "bytes_to_unicode"):
    tokenization_gpt2.bytes_to_unicode = lambda: GPT2_BYTE_TO_CHAR

from .config import load_config
from .counting import count_freq, count_recursive
from .model_ops import save_model
from .vocab_ops import (
    get_new_vocab_and_map,
    get_special_token_ids,
    load_tiktoken_vocab,
    reduce_to_target_size,
    save_vocab,
    save_vocab_tiktoken,
)
from .verification import smoke_test


def log(message: str, highlight: str | None = None) -> None:
    """Print a timestamped log message."""
    timestamp = colored(datetime.datetime.now().isoformat(), "green")
    if highlight:
        message = message.replace(highlight, colored(highlight, "green"))
    print(f"[{timestamp}]: {message}")


def main():
    parser = argparse.ArgumentParser(
        description="Prune LLM tokenizer vocabulary based on dataset usage"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config JSON file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model to prune",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for pruned model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL dataset for vocabulary analysis",
    )
    parser.add_argument(
        "--target_vocab_size",
        type=int,
        default=None,
        help="Target vocabulary size for lossy pruning (omit for lossless)",
    )
    parser.add_argument(
        "--inherit_vocab_count",
        type=str,
        default=None,
        help="Path to previous vocab counts to inherit",
    )
    parser.add_argument(
        "--no-smoke-test",
        action="store_true",
        help="Skip post-pruning smoke test",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create output directory
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        log(f"Creating output folder: {args.output_path}", args.output_path)

    # Load config and tokenizer (NOT the full model yet - that's 100GB+ for large models)
    log(f"Loading config and tokenizer from: {args.model_path}", args.model_path)
    model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    old_tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    model_type = getattr(model_config, "model_type", "unknown")
    log(f"Pruning vocabulary for: {colored(model_type, 'blue')}")

    # Get vocabulary sizes
    tokenizer_vocab_size = len(old_tokenizer.get_vocab())
    config_vocab_size = model_config.vocab_size

    # Handle vocab size mismatch
    if tokenizer_vocab_size > config_vocab_size:
        print(
            colored(
                f"Warning: Tokenizer has {tokenizer_vocab_size - config_vocab_size} more tokens "
                f"than model config allows ({tokenizer_vocab_size} > {config_vocab_size}). "
                f"Limiting to model config size.",
                "yellow",
            )
        )
        old_vocab_size = config_vocab_size
    elif tokenizer_vocab_size < config_vocab_size:
        print(
            f"Note: Model config vocab_size ({config_vocab_size}) includes "
            f"{config_vocab_size - tokenizer_vocab_size} padding/reserved slots. "
            f"Using actual tokenizer vocab size: {tokenizer_vocab_size}"
        )
        old_vocab_size = tokenizer_vocab_size
    else:
        old_vocab_size = tokenizer_vocab_size

    # Count token frequencies
    log(f"Loading dataset {args.dataset}", args.dataset)
    vocab_counts_path = os.path.join(args.output_path, "vocab_counts.torch")
    if os.path.exists(vocab_counts_path):
        vocab_counts = torch.load(vocab_counts_path, weights_only=False)
    else:
        vocab_counts = count_freq(
            data_path=args.dataset,
            vocab_size=old_vocab_size,
            tokenizer=old_tokenizer,
            inherit_vocab_count=args.inherit_vocab_count,
        )
        torch.save(vocab_counts, vocab_counts_path)

    # Get byte representations (limited to old_vocab_size)
    if config.tokenizer.type == "tiktoken":
        # For tiktoken, read raw bytes directly from the .model file
        old_bytes_list = load_tiktoken_vocab(args.model_path, max_token_id=old_vocab_size)
        log(f"Loaded {len(old_bytes_list)} tokens from tiktoken.model")
    else:
        # For BPE JSON, get vocab from tokenizer
        vocabulary = old_tokenizer.get_vocab()
        vocab_items = sorted(vocabulary.items(), key=lambda x: x[1])
        # Only include tokens within the valid vocab range
        vocab_items = [(token, idx) for token, idx in vocab_items if idx < old_vocab_size]
        old_bytes_list = [token.encode("utf-8") for token, _ in vocab_items]

    # Count recursive sub-token usage
    log("Calculating subword counts")
    recur_counts_path = os.path.join(args.output_path, "recur_counts.torch")
    if os.path.exists(recur_counts_path):
        recur_counts = torch.load(recur_counts_path, weights_only=False)
    else:
        recur_counts = count_recursive(
            vocab_size=old_vocab_size,
            vocab_counts=vocab_counts,
            old_bytes_list=old_bytes_list,
        )
        torch.save(recur_counts, recur_counts_path)

    # Reduce to target size if specified (lossy mode)
    if args.target_vocab_size is not None:
        log(
            f"Clipping vocabulary to new size: {args.target_vocab_size}",
            str(args.target_vocab_size),
        )
        vocab_counts, recur_counts = reduce_to_target_size(
            old_vocab_size=old_vocab_size,
            target_vocab_size=args.target_vocab_size,
            vocab_counts=vocab_counts,
            recur_counts=recur_counts,
            old_bytes_list=old_bytes_list,
        )

    # Detect special tokens
    log("Detecting special tokens to preserve")
    special_token_ids = get_special_token_ids(
        old_tokenizer, args.model_path, config, max_token_id=old_vocab_size
    )

    # Create new vocabulary
    log("Saving new vocabulary")
    new_bytes_list, mapping_new2old = get_new_vocab_and_map(
        old_bytes_list=old_bytes_list,
        old_vocab_size=old_vocab_size,
        vocab_counts=vocab_counts,
        recur_counts=recur_counts,
        special_token_ids=special_token_ids,
    )
    new_vocab_size = len(mapping_new2old)

    # Dispatch to correct save function based on tokenizer type
    if config.tokenizer.type == "tiktoken":
        save_vocab_tiktoken(
            new_bytes_list, mapping_new2old, args.output_path, args.model_path
        )
    else:
        save_vocab(new_bytes_list, mapping_new2old, args.output_path, args.model_path)

    # NOW load the model weights (only when we actually need them)
    log(f"Loading model weights from: {args.model_path}", args.model_path)
    old_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="cpu",
    )

    # Update and save model
    log("Updating checkpoint files")
    save_model(config, old_model, new_vocab_size, mapping_new2old, args.output_path)

    # Run smoke test unless disabled
    smoke_test_passed = True
    if not args.no_smoke_test:
        log("Running smoke test")
        smoke_test_passed = smoke_test(args.output_path)
        if not smoke_test_passed:
            print(
                colored(
                    "ERROR: Smoke test failed! The pruned model may not work correctly.",
                    "red",
                )
            )

    if smoke_test_passed:
        log(f"Pruning complete! Output saved to: {args.output_path}", args.output_path)
    else:
        log(
            f"Pruning finished with errors. Output saved to: {args.output_path}",
            args.output_path,
        )


if __name__ == "__main__":
    main()
