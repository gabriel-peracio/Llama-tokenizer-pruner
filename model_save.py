import os
import shutil
import torch


# def saving_updated_qwenvl(old_model, new_vocab_size, token_mapping, output_path):
#     # define new module
#     new_embeds = torch.nn.Embedding(
#         new_vocab_size,
#         old_model.config.hidden_size,
#         dtype=old_model.transformer.wte.weight.dtype,
#     )
#     new_lm_head = torch.nn.Linear(
#         old_model.config.hidden_size,
#         new_vocab_size,
#         bias=False,
#         dtype=old_model.lm_head.weight.dtype,
#     )
#     # get new module parameter from the old
#     assert len(set(token_mapping)) == new_vocab_size
#     new_embeds.weight.data = old_model.transformer.wte.weight.data[
#         torch.LongTensor(token_mapping, device=old_model.device)
#     ]
#     new_lm_head.weight.data = old_model.lm_head.weight.data[
#         torch.LongTensor(token_mapping, device=old_model.device)
#     ]
#     # update model
#     old_model.transformer.wte.weight = new_embeds.weight
#     old_model.lm_head.weight = new_lm_head.weight
#     old_model.transformer.wte.num_embeddings = new_vocab_size
#     old_model.lm_head.out_features = new_vocab_size
#     # update config
#     old_model.config.__dict__["vocab_size"] = new_vocab_size
#     old_model.config.__dict__["_name_or_path"] = output_path
#     old_model.config.__dict__["visual"]["image_start_id"] = token_mapping.index(
#         old_model.config.__dict__["visual"]["image_start_id"]
#     )
#     old_model.generation_config.__dict__["eos_token_id"] = token_mapping.index(
#         old_model.generation_config.__dict__["eos_token_id"]
#     )
#     # old_model.generation_config.__dict__["pad_token_id"] = token_mapping.index(
#     #     old_model.generation_config.__dict__["pad_token_id"]
#     # )
#     # save new model
#     print(f"Saving new model ckpt to {output_path}")
#     old_model.save_pretrained(output_path)


# def saving_updated_qwen(old_model, new_vocab_size, token_mapping, output_path):
#     # define new module
#     new_embeds = torch.nn.Embedding(
#         new_vocab_size,
#         old_model.config.hidden_size,
#         dtype=old_model.transformer.wte.weight.dtype,
#     )
#     new_lm_head = torch.nn.Linear(
#         old_model.config.hidden_size,
#         new_vocab_size,
#         bias=False,
#         dtype=old_model.lm_head.weight.dtype,
#     )
#     # get new module parameter from the old
#     assert len(set(token_mapping)) == new_vocab_size
#     new_embeds.weight.data = old_model.transformer.wte.weight.data[
#         torch.LongTensor(token_mapping, device=old_model.device)
#     ]
#     new_lm_head.weight.data = old_model.lm_head.weight.data[
#         torch.LongTensor(token_mapping, device=old_model.device)
#     ]
#     # update model
#     old_model.transformer.wte.weight = new_embeds.weight
#     old_model.lm_head.weight = new_lm_head.weight
#     old_model.transformer.wte.num_embeddings = new_vocab_size
#     old_model.lm_head.out_features = new_vocab_size
#     # update config
#     old_model.config.__dict__["vocab_size"] = new_vocab_size
#     old_model.config.__dict__["_name_or_path"] = output_path
#     old_model.generation_config.__dict__["eos_token_id"] = token_mapping.index(
#         old_model.generation_config.__dict__["eos_token_id"]
#     )
#     # old_model.generation_config.__dict__["pad_token_id"] = token_mapping.index(
#     #     old_model.generation_config.__dict__["pad_token_id"]
#     # )
#     # save new model
#     print(f"Saving new model ckpt to {output_path}")
#     old_model.save_pretrained(output_path)


def saving_updated_llama(old_model, new_vocab_size, token_mapping, output_path):
    # Create reverse mapping first
    old_to_new = {old: new for new, old in enumerate(token_mapping)}

    # define new modules
    new_embeds = torch.nn.Embedding(
        new_vocab_size,
        old_model.config.hidden_size,
        dtype=old_model.model.embed_tokens.weight.dtype,
    )
    new_lm_head = torch.nn.Linear(
        old_model.config.hidden_size,
        new_vocab_size,
        bias=False,
        dtype=old_model.lm_head.weight.dtype,
    )

    # get new module parameter from the old
    mapping_tensor = torch.LongTensor(token_mapping, device=old_model.device)
    new_embeds.weight.data = old_model.model.embed_tokens.weight.data[mapping_tensor]
    new_lm_head.weight.data = old_model.lm_head.weight.data[mapping_tensor]

    # update model
    old_model.model.embed_tokens = new_embeds
    old_model.lm_head = new_lm_head

    # update all configs before saving
    # 1. Model config
    old_model.config.vocab_size = new_vocab_size
    old_model.config._name_or_path = output_path
    old_model.config.bos_token_id = old_to_new[128000]
    old_model.config.eos_token_id = old_to_new[128001]
    old_model.config.pad_token_id = old_to_new[128004]

    # 2. Generation config
    old_model.generation_config.bos_token_id = old_to_new[128000]
    old_model.generation_config.eos_token_id = old_to_new[128001]
    old_model.generation_config.pad_token_id = old_to_new[128004]

    # save model (which will save all configs)
    print(f"Saving new model ckpt to {output_path}")
    old_model.save_pretrained(output_path, safe_serialization=True)


def saving_updated_qwen3(old_model, new_vocab_size, token_mapping, output_path):
    """
    Save Qwen3 model with pruned vocabulary.
    Note: Qwen3 has tie_word_embeddings=True, so lm_head shares weights with embed_tokens.
    """
    # Create reverse mapping first
    old_to_new = {old: new for new, old in enumerate(token_mapping)}

    # define new embedding module
    new_embeds = torch.nn.Embedding(
        new_vocab_size,
        old_model.config.hidden_size,
        dtype=old_model.model.embed_tokens.weight.dtype,
    )

    # get new module parameter from the old
    mapping_tensor = torch.LongTensor(token_mapping, device=old_model.device)
    new_embeds.weight.data = old_model.model.embed_tokens.weight.data[mapping_tensor]

    # update model
    old_model.model.embed_tokens = new_embeds

    # Update lm_head
    if hasattr(old_model, "lm_head"):
        if old_model.config.tie_word_embeddings:
            # If embeddings are tied, lm_head should use same weights as embed_tokens
            # We need to create a new lm_head that points to the new embeddings
            old_model.lm_head = torch.nn.Linear(
                old_model.config.hidden_size,
                new_vocab_size,
                bias=False,
                dtype=old_model.lm_head.weight.dtype,
            )
            # Tie the weights
            old_model.lm_head.weight = new_embeds.weight
        else:
            # If not tied, prune lm_head independently
            new_lm_head = torch.nn.Linear(
                old_model.config.hidden_size,
                new_vocab_size,
                bias=False,
                dtype=old_model.lm_head.weight.dtype,
            )
            new_lm_head.weight.data = old_model.lm_head.weight.data[mapping_tensor]
            old_model.lm_head = new_lm_head

    # update all configs before saving
    # 1. Model config
    old_model.config.vocab_size = new_vocab_size
    old_model.config._name_or_path = output_path

    # Map special token IDs to new indices (handle both single values and lists)
    def remap_token_id(token_id, old_to_new):
        """Helper to remap single token ID or list of token IDs"""
        if isinstance(token_id, list):
            return [old_to_new[tid] for tid in token_id if tid in old_to_new]
        elif token_id in old_to_new:
            return old_to_new[token_id]
        return token_id

    if (
        hasattr(old_model.config, "eos_token_id")
        and old_model.config.eos_token_id is not None
    ):
        old_model.config.eos_token_id = remap_token_id(
            old_model.config.eos_token_id, old_to_new
        )
    if (
        hasattr(old_model.config, "pad_token_id")
        and old_model.config.pad_token_id is not None
    ):
        old_model.config.pad_token_id = remap_token_id(
            old_model.config.pad_token_id, old_to_new
        )
    if (
        hasattr(old_model.config, "bos_token_id")
        and old_model.config.bos_token_id is not None
    ):
        old_model.config.bos_token_id = remap_token_id(
            old_model.config.bos_token_id, old_to_new
        )

    # 2. Generation config
    if hasattr(old_model, "generation_config"):
        if (
            hasattr(old_model.generation_config, "eos_token_id")
            and old_model.generation_config.eos_token_id is not None
        ):
            old_model.generation_config.eos_token_id = remap_token_id(
                old_model.generation_config.eos_token_id, old_to_new
            )
        if (
            hasattr(old_model.generation_config, "pad_token_id")
            and old_model.generation_config.pad_token_id is not None
        ):
            old_model.generation_config.pad_token_id = remap_token_id(
                old_model.generation_config.pad_token_id, old_to_new
            )
        if (
            hasattr(old_model.generation_config, "bos_token_id")
            and old_model.generation_config.bos_token_id is not None
        ):
            old_model.generation_config.bos_token_id = remap_token_id(
                old_model.generation_config.bos_token_id, old_to_new
            )

    # save model (which will save all configs)
    print(f"Saving new model ckpt to {output_path}")
    old_model.save_pretrained(output_path, safe_serialization=True)


def saving_updated_gemma3(old_model, new_vocab_size, token_mapping, output_path):
    # Create reverse mapping first
    old_to_new = {old: new for new, old in enumerate(token_mapping)}

    # define new modules
    new_embeds = torch.nn.Embedding(
        new_vocab_size,
        old_model.config.text_config.hidden_size,
        dtype=old_model.language_model.model.embed_tokens.weight.dtype,
    )

    # get new module parameter from the old
    mapping_tensor = torch.LongTensor(token_mapping, device=old_model.device)
    new_embeds.weight.data = old_model.language_model.model.embed_tokens.weight.data[
        mapping_tensor
    ]

    old_model.pop("language_model.lm_head.weight")

    # update model
    old_model.language_model.model.embed_tokens = new_embeds

    # update all configs before saving
    # 1. Model config
    old_model.config.text_config.vocab_size = new_vocab_size - 1 + 64
    # old_model.config.text_config._name_or_path = output_path

    old_model.config.boi_token_index = old_to_new[255999]
    old_model.config.eoi_token_index = old_to_new[256000]
    old_model.config.image_token_index = old_to_new[262144]

    old_model.config.bos_token_id = old_to_new[2]
    old_model.config.eos_token_id = old_to_new[1]
    old_model.config.pad_token_id = old_to_new[0]

    # 2. Generation config
    old_model.generation_config.bos_token_id = old_to_new[2]
    old_model.generation_config.eos_token_id = old_to_new[1]
    old_model.generation_config.pad_token_id = old_to_new[0]

    # save model (which will save all configs)
    print(f"Saving new model ckpt to {output_path}")
    old_model.save_pretrained(output_path, safe_serialization=True)
