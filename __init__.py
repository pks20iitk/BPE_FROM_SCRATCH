"""
tokenizers — Production-grade BPE tokenizer package.

Available tokenizers
--------------------
BasicTokenizer
    Byte-level BPE with no regex pre-splitting.  Best for learning.

RegexTokenizer
    Byte-level BPE with regex pre-splitting (GPT-2 / GPT-4 style).
    Use this for production training.

SentencePieceBPETokenizer
    BPE with SentencePiece space-normalisation convention (▁ markers).
    Compatible with LLaMA 1/2, Mistral v1/v2 vocabularies.

SentencePieceWrapper
    Thin wrapper around the official ``sentencepiece`` C++ library.
    Requires ``pip install sentencepiece``.

Constants
---------
GPT2_SPLIT_PATTERN   — regex used by GPT-2 / Llama 1-2
GPT4_SPLIT_PATTERN   — regex used by GPT-4 / Llama 3 / Mistral v3+
"""

from .base import Tokenizer, get_stats, merge, render_token
from .basic import BasicTokenizer
from .regex import RegexTokenizer, GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN
from .sentencepiece import SentencePieceBPETokenizer, SentencePieceWrapper

__all__ = [
    "Tokenizer",
    "BasicTokenizer",
    "RegexTokenizer",
    "SentencePieceBPETokenizer",
    "SentencePieceWrapper",
    "GPT2_SPLIT_PATTERN",
    "GPT4_SPLIT_PATTERN",
    "get_stats",
    "merge",
    "render_token",
]
