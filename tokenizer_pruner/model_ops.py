"""Model operations: loading, weight pruning, and saving."""

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

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
    try:
        model.save_pretrained(output_path, safe_serialization=True)
    except (AttributeError, TypeError) as e:
        # Fallback: save state dict directly (works around HF bugs with custom models)
        print(f"Warning: save_pretrained failed ({e}), using direct state_dict save")
        state_dict = model.state_dict()
        save_file(state_dict, os.path.join(output_path, "model.safetensors"))
        model.config.save_pretrained(output_path)


def save_model_direct(
    config: ModelConfig,
    model_path: str,
    new_vocab_size: int,
    token_mapping: list[int],
    output_path: str,
) -> None:
    """Save model with pruned vocabulary by directly editing safetensors.

    This avoids loading the full model into memory, making it practical
    for very large models (48B+).

    Args:
        config: Model configuration with architecture paths
        model_path: Path to original model
        new_vocab_size: New vocabulary size after pruning
        token_mapping: Mapping from new to old token indices
        output_path: Output directory
    """
    mapping_tensor = torch.LongTensor(token_mapping)

    # Find tensor names to modify
    embed_key = config.architecture.embedding_path.replace(".", ".") + ".weight"
    # Convert "model.embed_tokens" -> "model.embed_tokens.weight"

    lm_head_key = None
    if config.architecture.lm_head_path:
        lm_head_key = config.architecture.lm_head_path + ".weight"

    # Find all safetensors files
    model_path = Path(model_path)
    safetensor_files = list(model_path.glob("*.safetensors"))

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    # Check if there's an index file (sharded model)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        # Single file model
        weight_map = None

    # Find which files contain our target tensors
    embed_file = None
    lm_head_file = None

    if weight_map:
        embed_file = weight_map.get(embed_key)
        if lm_head_key:
            lm_head_file = weight_map.get(lm_head_key)
    else:
        # Single file - both are in the same file
        embed_file = safetensor_files[0].name
        lm_head_file = embed_file

    print(f"Embedding tensor '{embed_key}' in: {embed_file}")
    if lm_head_key:
        print(f"LM head tensor '{lm_head_key}' in: {lm_head_file}")

    # Process each safetensors file
    output_path = Path(output_path)
    new_weight_map = {}

    for sf_path in safetensor_files:
        sf_name = sf_path.name
        tensors = {}

        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # Prune embedding weights
                if key == embed_key:
                    print(f"Pruning {key}: {tensor.shape} -> ", end="")
                    tensor = tensor[mapping_tensor]
                    print(f"{tensor.shape}")

                # Prune lm_head weights
                elif key == lm_head_key:
                    print(f"Pruning {key}: {tensor.shape} -> ", end="")
                    tensor = tensor[mapping_tensor]
                    print(f"{tensor.shape}")

                tensors[key] = tensor
                new_weight_map[key] = sf_name

        # Save modified tensors
        out_sf_path = output_path / sf_name
        save_file(tensors, out_sf_path)
        print(f"Saved {out_sf_path}")

    # Update and save index file if it exists
    if index_path.exists():
        new_index = {
            "metadata": index.get("metadata", {}),
            "weight_map": new_weight_map,
        }
        with open(output_path / "model.safetensors.index.json", "w") as f:
            json.dump(new_index, f, indent=2)
        print("Saved model.safetensors.index.json")
