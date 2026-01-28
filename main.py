import os
import argparse
from transformers import AutoTokenizer, Qwen3ForCausalLM
from vocab_count import count_freq, count_recursive
from vocab_save import (
    get_new_vocab_and_map,
    save_vocab,
    reduce_to_target_size,
    get_special_token_ids,
)
from model_save import *
from termcolor import colored
import datetime


def main():
    # start vocabulary pruning
    print(
        f"[{colored(datetime.datetime.now().isoformat(), 'green')}]: Begin Pruning Vocabulary for: {colored('Qwen 3', 'blue')}"
    )

    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--inherit_vocab_count", type=str, default=None)
    parser.add_argument("--target_vocab_size", type=int, default=None)
    args = parser.parse_args()

    # valid check
    assert args.dataset is not None, "Must provide Support Data."

    # init output path
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print(
            f"[{colored(datetime.datetime.now().isoformat(), 'green')}]: Creating output folder: {colored(args.output_path, 'green')}"
        )

    # load old model and tokenizer
    print(
        f"[{colored(datetime.datetime.now().isoformat(), 'green')}]: Loading model and tokenizer from: {colored(args.model_path, 'green')}"
    )
    old_model = Qwen3ForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, device_map="cpu"
    )
    old_tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    # Use actual tokenizer vocab size (not config which may include padding)
    old_vocab_size = len(old_tokenizer.get_vocab())
    config_vocab_size = old_model.config.__dict__["vocab_size"]

    if old_vocab_size > config_vocab_size:
        raise ValueError(
            f"Tokenizer has more tokens than config allows: {old_vocab_size} > {config_vocab_size}"
        )

    if old_vocab_size != config_vocab_size:
        print(
            f"Note: Model config vocab_size ({config_vocab_size}) includes "
            f"{config_vocab_size - old_vocab_size} padding/reserved slots. "
            f"Using actual tokenizer vocab size: {old_vocab_size}"
        )

    # using support data
    if args.dataset is not None:
        print(
            f"[{colored(datetime.datetime.now().isoformat(), 'green')}]: Loading dataset {colored(args.dataset, 'green')}"
        )
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
    else:
        vocab_counts = [0 for _ in range(old_vocab_size)]

    vocabulary = old_tokenizer.get_vocab()
    vocab_items = sorted(vocabulary.items(), key=lambda x: x[1])
    old_bytes_list = [token.encode("utf-8") for token, _ in vocab_items]

    # # load bpe file
    # tiktoken_bpe_file = get_bpe_file(args.model_path)
    # print(f"==> Load tiktoken bpe file from: {tiktoken_bpe_file}")
    # with open(tiktoken_bpe_file, "rb") as f:
    #     contents = f.read()
    # old_bytes_list = [base64.b64decode(token) for token, rank in (line.split() for line in contents.splitlines() if line)]

    # sub-token count
    print(
        f"[{colored(datetime.datetime.now().isoformat(), 'green')}]: Calculating subword counts"
    )
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

    # reduce vocab to target size
    if args.target_vocab_size is not None:
        print(
            f"[{colored(datetime.datetime.now().isoformat(), 'green')}]: Clipping vocabulary to new size: {colored(args.target_vocab_size, 'green')}"
        )
        vocab_counts, recur_counts = reduce_to_target_size(
            old_vocab_size=old_vocab_size,
            target_vocab_size=args.target_vocab_size,
            vocab_counts=vocab_counts,
            recur_counts=recur_counts,
            old_bytes_list=old_bytes_list,
        )

    # detect special tokens to preserve
    print(
        f"[{colored(datetime.datetime.now().isoformat(), 'green')}]: Detecting special tokens to preserve"
    )
    special_token_ids = get_special_token_ids(old_tokenizer, args.model_path)

    # get new vocab
    print(
        f"[{colored(datetime.datetime.now().isoformat(), 'green')}]: Saving new sentencepiece vocabulary"
    )
    new_bytes_list, mapping_new2old = get_new_vocab_and_map(
        old_bytes_list=old_bytes_list,
        old_vocab_size=old_vocab_size,
        vocab_counts=vocab_counts,
        recur_counts=recur_counts,
        special_token_ids=special_token_ids,
    )
    new_vocab_size = len(mapping_new2old)
    save_vocab(new_bytes_list, mapping_new2old, args.output_path, args.model_path)

    # update model ckpt
    print(
        f"[{colored(datetime.datetime.now().isoformat(), 'green')}]: Updating checkpoint files"
    )
    saving_updated_qwen3(old_model, new_vocab_size, mapping_new2old, args.output_path)


if __name__ == "__main__":
    main()
