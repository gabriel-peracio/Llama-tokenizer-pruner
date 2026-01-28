"""Verification utilities for pruned tokenizers and models."""

import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .counting import get_text_list

BATCH_SIZE = max(1, os.cpu_count() - 1)


def verify_tokenization(
    old_model_path: str,
    new_model_path: str,
    dataset_path: str,
) -> tuple[int, list[str]]:
    """Verify that old and new tokenizers produce equivalent results.

    Args:
        old_model_path: Path to original model
        new_model_path: Path to pruned model
        dataset_path: Path to test dataset (JSONL)

    Returns:
        Tuple of (mismatch_count, mismatch_examples)
    """
    print(f"Load old tokenizer from {old_model_path}")
    old_tokenizer = AutoTokenizer.from_pretrained(
        old_model_path, trust_remote_code=True
    )

    print(f"Load new tokenizer from {new_model_path}")
    new_tokenizer = AutoTokenizer.from_pretrained(
        new_model_path, trust_remote_code=True
    )

    print(
        f"Load token mapping from {os.path.join(new_model_path, 'token_mapping.torch')}"
    )
    mapping_new2old = (
        torch.load(
            os.path.join(new_model_path, "token_mapping.torch"),
            map_location="cpu",
            weights_only=False,
        )
        .long()
        .tolist()
    )

    print(f"Load check data from {dataset_path}")
    prompt_list = get_text_list(dataset_path)

    print("Verifying tokenization consistency...")
    mismatch_list = []

    for i in tqdm(range(0, len(prompt_list), BATCH_SIZE)):
        batch_prompts = prompt_list[i : i + BATCH_SIZE]

        old_context_tokens_list = old_tokenizer(batch_prompts)
        new_context_tokens_list = new_tokenizer(batch_prompts)

        for prompt, old_context_tokens, new_context_tokens in zip(
            batch_prompts,
            old_context_tokens_list["input_ids"],
            new_context_tokens_list["input_ids"],
        ):
            if len(old_context_tokens) != len(new_context_tokens):
                mismatch_list.append(prompt)
                continue

            if not all(
                old_token == mapping_new2old[new_token]
                for old_token, new_token in zip(old_context_tokens, new_context_tokens)
            ):
                mismatch_list.append(prompt)

    if len(mismatch_list) == 0:
        print(f"==> All {len(prompt_list)} samples tokenize correctly!")
    else:
        print(f"==> Mismatch: {len(mismatch_list)} / {len(prompt_list)} samples")
        if len(mismatch_list) > 0:
            old_context_tokens = old_tokenizer.encode(mismatch_list[0])
            print(f"==> Mismatch example 0 old tokens: {old_context_tokens}")
            new_context_tokens = new_tokenizer.encode(mismatch_list[0])
            mapped_tokens = [mapping_new2old[t] for t in new_context_tokens]
            print(f"==> Mismatch example 0 new tokens (mapped): {mapped_tokens}")

    return len(mismatch_list), mismatch_list


def smoke_test(
    model_path: str,
    test_string: str = "Hello, world!",
) -> bool:
    """Run a smoke test on a pruned model.

    Loads the model, tokenizes a test string, runs a forward pass,
    and verifies the output shape matches the new vocab size.

    Args:
        model_path: Path to the pruned model
        test_string: String to test with

    Returns:
        True if smoke test passes, False otherwise
    """
    print(f"Running smoke test on {model_path}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
        )

        # Tokenize test string
        inputs = tokenizer(test_string, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Run forward pass
        with torch.no_grad():
            outputs = model(input_ids)

        # Verify output shape
        logits = outputs.logits
        vocab_size = model.config.vocab_size
        actual_vocab_size = logits.shape[-1]

        if actual_vocab_size != vocab_size:
            print(
                f"==> FAILED: Output vocab size ({actual_vocab_size}) doesn't match "
                f"config vocab size ({vocab_size})"
            )
            return False

        print(f"==> PASSED: Model forward pass successful")
        print(f"    Input shape: {input_ids.shape}")
        print(f"    Output shape: {logits.shape}")
        print(f"    Vocab size: {vocab_size}")

        return True

    except Exception as e:
        print(f"==> FAILED: {e}")
        return False


def main():
    """CLI entry point for standalone verification."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify pruned tokenizer/model")
    parser.add_argument("--old_model_path", type=str, required=True)
    parser.add_argument("--new_model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()

    mismatch_count, _ = verify_tokenization(
        args.old_model_path,
        args.new_model_path,
        args.dataset,
    )

    if args.smoke_test:
        smoke_test(args.new_model_path)

    return mismatch_count == 0


if __name__ == "__main__":
    exit(0 if main() else 1)
