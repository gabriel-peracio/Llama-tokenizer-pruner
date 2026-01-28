"""Configuration dataclass and loader for tokenizer pruning."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ArchitectureConfig:
    """Model architecture paths for embedding and lm_head layers."""

    embedding_path: str
    lm_head_path: Optional[str] = None  # None if tied/fused with embeddings


@dataclass
class TokenizerConfig:
    """Tokenizer format configuration."""

    type: str  # "bpe_json", "tiktoken", "sentencepiece"
    byte_fallback_pattern: str = r"<0x[0-9A-Fa-f]{2}>"


@dataclass
class PreserveTokensConfig:
    """Configuration for tokens to always preserve."""

    patterns: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Complete configuration for a model family."""

    model_family: str
    architecture: ArchitectureConfig
    tokenizer: TokenizerConfig
    preserve_tokens: PreserveTokensConfig = field(
        default_factory=lambda: PreserveTokensConfig()
    )

    def get_preserve_patterns(self) -> list[re.Pattern]:
        """Compile preserve patterns to regex objects."""
        return [re.compile(p) for p in self.preserve_tokens.patterns]

    def get_byte_fallback_pattern(self) -> re.Pattern:
        """Compile byte fallback pattern to regex object."""
        return re.compile(self.tokenizer.byte_fallback_pattern)


def load_config(config_path: str | Path) -> ModelConfig:
    """Load and validate a model configuration from a JSON file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = json.load(f)

    # Validate required fields
    required_fields = ["model_family", "architecture", "tokenizer"]
    for field_name in required_fields:
        if field_name not in data:
            raise ValueError(f"Config missing required field: {field_name}")

    if "embedding_path" not in data["architecture"]:
        raise ValueError("Config architecture missing required field: embedding_path")

    if "type" not in data["tokenizer"]:
        raise ValueError("Config tokenizer missing required field: type")

    valid_tokenizer_types = ["bpe_json", "tiktoken", "sentencepiece"]
    if data["tokenizer"]["type"] not in valid_tokenizer_types:
        raise ValueError(
            f"Invalid tokenizer type: {data['tokenizer']['type']}. "
            f"Must be one of: {valid_tokenizer_types}"
        )

    # Build config objects
    architecture = ArchitectureConfig(
        embedding_path=data["architecture"]["embedding_path"],
        lm_head_path=data["architecture"].get("lm_head_path"),
    )

    tokenizer = TokenizerConfig(
        type=data["tokenizer"]["type"],
        byte_fallback_pattern=data["tokenizer"].get(
            "byte_fallback_pattern", r"<0x[0-9A-Fa-f]{2}>"
        ),
    )

    preserve_tokens = PreserveTokensConfig(
        patterns=data.get("preserve_tokens", {}).get("patterns", [])
    )

    return ModelConfig(
        model_family=data["model_family"],
        architecture=architecture,
        tokenizer=tokenizer,
        preserve_tokens=preserve_tokens,
    )
