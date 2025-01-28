import os
import torch
import argparse
from transformers import AutoTokenizer
from vocab_count import get_text_list
from tqdm import tqdm

BATCH_SIZE = os.cpu_count() - 1


def main():
    # start vocabulary pruning
    print("============ Start Llama Vocabulary Pruning ==========")

    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_model_path", type=str, default=None)
    parser.add_argument("--new_model_path", type=str, default=None)
    parser.add_argument("--support_data", type=str, default=None)
    args = parser.parse_args()

    # load tokenziers
    print(f"Load old tokenizer from {args.old_model_path}")
    old_tokenizer = AutoTokenizer.from_pretrained(
        args.old_model_path, trust_remote_code=True
    )
    print(f"Load new tokenizer from {args.new_model_path}")
    new_tokenizer = AutoTokenizer.from_pretrained(
        args.new_model_path, trust_remote_code=True
    )
    print(
        f"Load token mapping from {os.path.join(args.new_model_path, 'token_mapping.torch')}"
    )
    mapping_new2old = (
        torch.load(
            os.path.join(args.new_model_path, "token_mapping.torch"),
            map_location="cpu",
            weights_only=False,
        )
        .long()
        .tolist()
    )

    # load data
    print(f"Load check data from {args.support_data}")
    prompt_list = get_text_list(args.support_data)

    # check prompt
    print(f"For plain text list that doesn't require system prompt")
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
                break  # Break early, since we only print the first example anyway
            elif not all(
                [
                    old_token == mapping_new2old[new_token]
                    for old_token, new_token in zip(
                        old_context_tokens, new_context_tokens
                    )
                ]
            ):
                mismatch_list.append(prompt)
                break  # Break early, since we only print the first example anyway

    print(f"==> Mismatch num in plain text list: {len(mismatch_list)}. All Correct!")
    if len(mismatch_list) > 0:
        old_context_tokens = old_tokenizer.encode(mismatch_list[0])
        print(f"==> Mismatch example 0 old tokens: {old_context_tokens}")
        new_context_tokens = new_tokenizer.encode(mismatch_list[0])
        new_context_tokens = [
            mapping_new2old[new_token] for new_token in new_context_tokens
        ]
        print(f"==> Mismatch example 0 new tokens: {new_context_tokens}")


if __name__ == "__main__":
    main()
