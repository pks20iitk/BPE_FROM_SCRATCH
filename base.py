"""
base.py — Abstract base class for all tokenizers in this project.

Defines the train / encode / decode interface and provides shared
utility functions (get_stats, merge) that every concrete tokenizer
inherits.  Also implements save / load so trained models can be
persisted as a pair of files:
  <name>.model  — machine-readable (merges + special tokens, JSON)
  <name>.vocab  — human-readable vocabulary listing

Influenced by:
  • Karpathy's minBPE  (https://github.com/karpathy/minbpe)
  • Raschka's BPE from scratch (https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------

def get_stats(
    ids: List[int],
    counts: Optional[Dict[Tuple[int, int], int]] = None,
) -> Dict[Tuple[int, int], int]:
    """Count every adjacent pair (bigram) in *ids*.

    Parameters
    ----------
    ids:
        Flat list of integer token IDs.
    counts:
        Existing counter to accumulate into.  If *None* a new one is created.

    Returns
    -------
    dict mapping (id_a, id_b) → occurrence count.
    """
    counts = counts or {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replace every occurrence of *pair* in *ids* with *new_id*.

    This is the core in-place compression step of BPE.

    Parameters
    ----------
    ids:
        Input token list.
    pair:
        The (a, b) bigram to collapse.
    new_id:
        The replacement token ID.

    Returns
    -------
    New list with all occurrences of *pair* replaced.
    """
    result: List[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


def render_token(t: bytes) -> str:
    """Return a printable representation of a raw byte-string token."""
    s = t.decode("utf-8", errors="replace")
    return "".join(
        ch if ch.isprintable() and ch not in {" ", "\n", "\r", "\t"} else f"\\x{ord(ch):02x}"
        for ch in s
    )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Tokenizer(ABC):
    """Abstract base for all tokenizers.

    Subclasses must implement :meth:`train`, :meth:`encode`, and
    :meth:`decode`.  Save / load and the utility helpers live here so
    they are automatically available to every subclass.

    Attributes
    ----------
    merges:
        dict[(int, int) → int] — the learned BPE merge table.
    special_tokens:
        dict[str → int] — special tokens added to the vocabulary.
    vocab:
        dict[int → bytes] — maps token ID to raw bytes.
    """

    def __init__(self) -> None:
        self.merges: Dict[Tuple[int, int], int] = {}
        self.special_tokens: Dict[str, int] = {}
        self.vocab: Dict[int, bytes] = self._build_initial_vocab()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_initial_vocab() -> Dict[int, bytes]:
        """Seed vocabulary: token 0–255 map to their single raw byte."""
        return {i: bytes([i]) for i in range(256)}

    def _build_vocab_from_merges(self) -> None:
        """Reconstruct :attr:`vocab` from :attr:`merges` and :attr:`special_tokens`."""
        self.vocab = self._build_initial_vocab()
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode("utf-8")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn BPE merges from *text* until vocabulary reaches *vocab_size*."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of integer token IDs."""

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to a string."""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, file_prefix: str) -> None:
        """Save tokenizer state to ``<file_prefix>.model`` and ``<file_prefix>.vocab``.

        The ``.model`` file is machine-readable JSON.
        The ``.vocab`` file is a human-readable listing.
        """
        model_path = file_prefix + ".model"
        vocab_path = file_prefix + ".vocab"

        # --- .model (JSON) ---
        model_data = {
            "merges": [[list(pair), idx] for pair, idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)

        # --- .vocab (human-readable) ---
        inverted_special = {v: k for k, v in self.special_tokens.items()}
        with open(vocab_path, "w", encoding="utf-8") as f:
            for idx, token_bytes in self.vocab.items():
                token_str = render_token(token_bytes)
                if idx in inverted_special:
                    token_str = inverted_special[idx]
                f.write(f"[{idx:6d}] {token_str}\n")

        print(f"Saved: {model_path}, {vocab_path}")

    def load(self, model_path: str) -> None:
        """Restore tokenizer state from a ``.model`` file.

        Parameters
        ----------
        model_path:
            Path produced by :meth:`save`.
        """
        assert model_path.endswith(".model"), "Expected a .model file"
        with open(model_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        self.merges = {
            (pair[0], pair[1]): idx
            for pair, idx in model_data["merges"]
        }
        self.special_tokens = model_data.get("special_tokens", {})
        self._build_vocab_from_merges()
        print(f"Loaded: {model_path} | vocab size = {len(self.vocab)}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the current vocabulary."""
        return len(self.vocab)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"vocab_size={self.vocab_size}, "
            f"merges={len(self.merges)}, "
            f"special_tokens={list(self.special_tokens.keys())})"
        )
