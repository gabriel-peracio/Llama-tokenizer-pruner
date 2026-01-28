"""
Tokenizer Pruner - A modular, config-driven package for pruning LLM tokenizers.

Supports multiple model architectures and tokenizer formats.
"""

__version__ = "0.1.0"

from .config import ModelConfig, load_config
from .cli import main

__all__ = ["ModelConfig", "load_config", "main", "__version__"]
