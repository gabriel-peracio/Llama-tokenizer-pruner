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
