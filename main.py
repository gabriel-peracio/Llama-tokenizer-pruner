import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from vocab_count import count_freq, count_recursive
from vocab_save import get_new_vocab_and_map, save_vocab, reduce_to_target_size
from model_save import *


def main():
    # start vocabulary pruning
    print("============ Start LLaMA Vocabulary Pruning ==========")

    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_model_path", type=str, default=None)
    parser.add_argument("--new_model_path", type=str, default=None)
    parser.add_argument("--support_data", type=str, default=None)
    parser.add_argument("--inherit_vocab_count", type=str, default=None)
    parser.add_argument("--target_size", type=int, default=None)
    args = parser.parse_args()

    # valid check
    assert args.support_data is not None, "Must provide Support Data."

    # init output path
    if not os.path.exists(args.new_model_path):
        os.makedirs(args.new_model_path)
        print(f"==> Create output folder: {args.new_model_path}")

    # load old model and tokenizer
    print(f"==> Load old model and tokenizer from: {args.old_model_path}")
    old_model = AutoModelForCausalLM.from_pretrained(
        args.old_model_path, trust_remote_code=True, device_map="cpu"
    )
    old_tokenizer = AutoTokenizer.from_pretrained(
        args.old_model_path, trust_remote_code=True
    )
    old_vocab_size = old_model.config.__dict__["vocab_size"]
    print(f"Tokenizer has vocabulary size {old_vocab_size}")

    # using support data
    if args.support_data is not None:
        print(f"==> Loading Support Data (for Freqs Count) from: {args.support_data}")
        vocab_counts_path = os.path.join(args.new_model_path, "vocab_counts.torch")
        if os.path.exists(vocab_counts_path):
            vocab_counts = torch.load(vocab_counts_path, weights_only=False)
        else:
            vocab_counts = count_freq(
                data_path=args.support_data,
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
    # tiktoken_bpe_file = get_bpe_file(args.old_model_path)
    # print(f"==> Load tiktoken bpe file from: {tiktoken_bpe_file}")
    # with open(tiktoken_bpe_file, "rb") as f:
    #     contents = f.read()
    # old_bytes_list = [base64.b64decode(token) for token, rank in (line.split() for line in contents.splitlines() if line)]

    # sub-token count
    print(f"==> Recursively calculate sub-token count")
    recur_counts_path = os.path.join(args.new_model_path, "recur_counts.torch")
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
    if args.target_size is not None:
        print(f"==> Reduce vocab to the target size {args.target_size}")
        vocab_counts, recur_counts = reduce_to_target_size(
            old_vocab_size=old_vocab_size,
            target_vocab_size=args.target_size,
            vocab_counts=vocab_counts,
            recur_counts=recur_counts,
            old_bytes_list=old_bytes_list,
        )

    # get new vocab
    print(f"==> Get new vocabulary bpe file and save it")
    new_bytes_list, mapping_new2old = get_new_vocab_and_map(
        old_bytes_list=old_bytes_list,
        old_vocab_size=old_vocab_size,
        vocab_counts=vocab_counts,
        recur_counts=recur_counts,
    )
    new_vocab_size = len(mapping_new2old)
    save_vocab(
        new_bytes_list, mapping_new2old, args.new_model_path, args.old_model_path
    )

    # update model ckpt
    print(f"==> Update model ckpt for new tokenizer")
    saving_updated_llama(
        old_model, new_vocab_size, mapping_new2old, args.new_model_path
    )


if __name__ == "__main__":
    main()
