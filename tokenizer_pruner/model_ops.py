"""Model operations: loading, weight pruning, and saving."""

import torch

from .config import ModelConfig


def _get_nested_attr(obj, path: str):
    """Get a nested attribute using dot notation.

    Args:
        obj: Object to traverse
        path: Dot-separated path (e.g., "model.embed_tokens")

    Returns:
        The attribute at the specified path
    """
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_nested_attr(obj, path: str, value):
    """Set a nested attribute using dot notation.

    Args:
        obj: Object to traverse
        path: Dot-separated path (e.g., "model.embed_tokens")
        value: Value to set
    """
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _remap_token_id(token_id, old_to_new: dict):
    """Remap single token ID or list of token IDs.

    Args:
        token_id: Single ID or list of IDs
        old_to_new: Mapping from old to new indices

    Returns:
        Remapped ID(s)
    """
    if isinstance(token_id, list):
        return [old_to_new[tid] for tid in token_id if tid in old_to_new]
    elif token_id in old_to_new:
        return old_to_new[token_id]
    return token_id


def save_model(
    config: ModelConfig,
    model,
    new_vocab_size: int,
    token_mapping: list[int],
    output_path: str,
) -> None:
    """Save model with pruned vocabulary using config-based paths.

    This is a generic implementation that uses the embedding_path and lm_head_path
    from the config to locate and update the relevant layers.

    Args:
        config: Model configuration with architecture paths
        model: The loaded model
        new_vocab_size: New vocabulary size after pruning
        token_mapping: Mapping from new to old token indices
        output_path: Output directory
    """
    old_to_new = {old: new for new, old in enumerate(token_mapping)}
    mapping_tensor = torch.LongTensor(token_mapping)

    # Get embedding layer using config path
    embedding_path = config.architecture.embedding_path
    embed_tokens = _get_nested_attr(model, embedding_path)

    # Create new embedding layer
    new_embeds = torch.nn.Embedding(
        new_vocab_size,
        embed_tokens.weight.shape[1],  # hidden_size
        dtype=embed_tokens.weight.dtype,
    )
    new_embeds.weight.data = embed_tokens.weight.data[mapping_tensor]

    # Update embedding layer
    _set_nested_attr(model, embedding_path, new_embeds)

    # Handle lm_head
    lm_head_path = config.architecture.lm_head_path
    if lm_head_path is not None:
        # lm_head exists and is specified
        lm_head = _get_nested_attr(model, lm_head_path)

        # Check if embeddings are tied
        tie_embeddings = getattr(model.config, "tie_word_embeddings", False)

        if tie_embeddings:
            # Create new lm_head with tied weights
            new_lm_head = torch.nn.Linear(
                lm_head.weight.shape[1],  # hidden_size
                new_vocab_size,
                bias=False,
                dtype=lm_head.weight.dtype,
            )
            # Tie to embeddings
            new_lm_head.weight = new_embeds.weight
        else:
            # Prune lm_head independently
            new_lm_head = torch.nn.Linear(
                lm_head.weight.shape[1],  # hidden_size
                new_vocab_size,
                bias=False,
                dtype=lm_head.weight.dtype,
            )
            new_lm_head.weight.data = lm_head.weight.data[mapping_tensor]

        _set_nested_attr(model, lm_head_path, new_lm_head)

    # Update model config
    model.config.vocab_size = new_vocab_size
    model.config._name_or_path = output_path

    # Remap special token IDs in config
    for attr in ["eos_token_id", "pad_token_id", "bos_token_id"]:
        if hasattr(model.config, attr) and getattr(model.config, attr) is not None:
            old_val = getattr(model.config, attr)
            new_val = _remap_token_id(old_val, old_to_new)
            setattr(model.config, attr, new_val)

    # Update generation config if present
    if hasattr(model, "generation_config"):
        for attr in ["eos_token_id", "pad_token_id", "bos_token_id"]:
            if (
                hasattr(model.generation_config, attr)
                and getattr(model.generation_config, attr) is not None
            ):
                old_val = getattr(model.generation_config, attr)
                new_val = _remap_token_id(old_val, old_to_new)
                setattr(model.generation_config, attr, new_val)

    # Save model
    print(f"Saving new model ckpt to {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
