"""
regex.py — RegexTokenizer

A byte-level BPE tokenizer that pre-splits text with a regex pattern
before running BPE.  This prevents merges from crossing linguistic
boundaries (letters ↔ numbers ↔ punctuation), which is the key
improvement introduced in the GPT-2 paper and retained in GPT-4.

Influenced by:
  • Karpathy's minBPE  (https://github.com/karpathy/minbpe)
  • Raschka's BPE from scratch (https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)

Key idea
--------
Instead of treating the entire text as one byte stream, we first split
it into *chunks* using a regex (e.g. separate words, numbers, spaces,
punctuation).  BPE merges are then learned and applied independently
within each chunk — so "hello123" can never produce a merge that spans
the word/number boundary.

Vocabulary layout (GPT-4 / cl100k_base example)
-------------------------------------------------
  0–255       raw bytes
  256 …       BPE merges (learned pairs)
  100256 …    special tokens, e.g. <|endoftext|>

Quick start
-----------
>>> from tokenizers.regex import RegexTokenizer
>>> tok = RegexTokenizer()
>>> tok.train(open("corpus.txt").read(), vocab_size=512)
>>> ids = tok.encode("Hello, world!", allowed_special=set())
>>> tok.decode(ids)
'Hello, world!'
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Set, Tuple

from .base import Tokenizer, get_stats, merge

# ---------------------------------------------------------------------------
# Canonical regex patterns
# ---------------------------------------------------------------------------

# GPT-2 pattern — used for GPT-2 and Llama 1/2
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)|"""   # contractions
    r"""[^\r\n\p{L}\p{N}]?\p{L}+|"""  # words (optionally prefixed by non-alphanum)
    r"""\p{N}{1,3}|"""             # numbers (at most 3 digits)
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|"""  # punctuation / symbols
    r"""\s*[\r\n]+|"""             # newlines
    r"""\s+(?!\S)|"""              # trailing whitespace
    r"""\s+"""                     # other whitespace
)

# GPT-4 pattern — cl100k_base, also used by Llama 3, Mistral v3+
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|"""
    r"""[^\r\n\p{L}\p{N}]?+\p{L}+|"""
    r"""\p{N}{1,3}|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|"""
    r"""\s*[\r\n]|"""
    r"""\s+(?!\S)|"""
    r"""\s+"""
)


def _compile(pattern: str) -> re.Pattern:
    """Compile *pattern* with Unicode flag; try the ``regex`` package first."""
    try:
        import regex  # faster, required for \p{L} etc.
        return regex.compile(pattern)
    except ImportError:
        # Fallback: strip \p{} classes and use stdlib re (partial support)
        import re as _re
        simplified = re.sub(r"\\p\{[^}]+\}", r"\\w", pattern)
        return _re.compile(simplified)


class RegexTokenizer(Tokenizer):
    """BPE tokenizer with regex pre-splitting and special-token support.

    Parameters
    ----------
    pattern:
        Split regex.  Defaults to the GPT-4 pattern.  Use
        ``GPT2_SPLIT_PATTERN`` for GPT-2 compatibility.
    """

    def __init__(self, pattern: Optional[str] = None) -> None:
        super().__init__()
        self._pattern_str = pattern or GPT4_SPLIT_PATTERN
        self._pattern = _compile(self._pattern_str)
        # byte_shuffle maps raw byte values to token IDs (identity by default)
        self._byte_shuffle: Dict[int, int] = {}
        self._inverse_byte_shuffle: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn BPE merges from *text* using regex pre-splitting.

        Parameters
        ----------
        text:
            Training corpus.
        vocab_size:
            Target vocabulary size (≥ 256).
        verbose:
            Print progress.
        """
        assert vocab_size >= 256, "vocab_size must be ≥ 256"
        num_merges = vocab_size - 256

        # 1. Pre-split text into chunks
        chunks = re.findall(self._pattern, text)
        # 2. Encode each chunk to byte IDs independently
        ids_list: List[List[int]] = [list(ch.encode("utf-8")) for ch in chunks]

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        t0 = time.time()
        for i in range(num_merges):
            # Accumulate stats across *all* chunks
            stats: Dict[Tuple[int, int], int] = {}
            for ids in ids_list:
                get_stats(ids, stats)

            if not stats:
                if verbose:
                    print(f"[Merge {i+1}/{num_merges}] No pairs — stopping early.")
                break

            best_pair = max(stats, key=stats.get)
            new_id = 256 + i

            ids_list = [merge(ids, best_pair, new_id) for ids in ids_list]
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose:
                elapsed = time.time() - t0
                tok_str = self.vocab[new_id].decode("utf-8", errors="replace")
                total = sum(len(ids) for ids in ids_list)
                print(
                    f"  merge {i+1:4d}/{num_merges} "
                    f"({best_pair[0]}, {best_pair[1]}) → {new_id}  "
                    f"'{tok_str}'  freq={stats[best_pair]}  "
                    f"total_tokens={total}  elapsed={elapsed:.1f}s"
                )

        if verbose:
            total_after = sum(len(ids) for ids in ids_list)
            original = sum(len(ch.encode("utf-8")) for ch in chunks)
            print(
                f"\nTraining complete: {len(self.merges)} merges, "
                f"vocab={self.vocab_size}, "
                f"compression={original}/{total_after}="
                f"{original/max(total_after,1):.2f}x"
            )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def register_special_tokens(self, special_tokens: Dict[str, int]) -> None:
        """Register special tokens (e.g. ``{"<|endoftext|>": 100257}``).

        Special tokens are never split by BPE — they are matched verbatim
        before any other processing.
        """
        self.special_tokens = dict(special_tokens)
        # Update vocab
        for token, idx in special_tokens.items():
            self.vocab[idx] = token.encode("utf-8")

    def encode_ordinary(self, text: str) -> List[int]:
        """Encode *text* without any special-token handling.

        Splits text by the stored regex pattern, encodes each chunk
        to bytes, then applies BPE merges in rank order.
        """
        chunks = re.findall(self._pattern, text)
        ids: List[int] = []
        for chunk in chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            chunk_ids = self._apply_merges(chunk_ids)
            ids.extend(chunk_ids)
        return ids

    def encode(
        self,
        text: str,
        allowed_special: str | Set[str] = "none_raise",
    ) -> List[int]:
        """Encode *text* with optional special-token support.

        Parameters
        ----------
        text:
            Input string.
        allowed_special:
            Controls how special tokens in *text* are handled.

            * ``"all"``        — all registered special tokens are tokenised.
            * ``"none"``       — special tokens in input are silently ignored.
            * ``"none_raise"`` — raise ``ValueError`` if any special token appears.
            * ``set``          — only the listed tokens are tokenised.

        Returns
        -------
        List of integer token IDs.
        """
        if not self.special_tokens:
            return self.encode_ordinary(text)

        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            for tok in self.special_tokens:
                if tok in text:
                    raise ValueError(
                        f"Special token {tok!r} found in text. "
                        "Pass allowed_special='all' or a set of allowed tokens."
                    )
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"Unrecognised allowed_special value: {allowed_special!r}")

        if not special:
            return self.encode_ordinary(text)

        # Build a regex that matches any of the allowed special tokens
        special_pattern = "(" + "|".join(re.escape(s) for s in sorted(special, key=len, reverse=True)) + ")"
        parts = re.split(special_pattern, text)
        ids: List[int] = []
        for part in parts:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to a string.

        Parameters
        ----------
        ids:
            Token ID list produced by :meth:`encode`.

        Returns
        -------
        UTF-8 decoded string (replacement character for invalid bytes).
        """
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            else:
                raise ValueError(f"Token ID {i} not in vocabulary.")
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_merges(self, ids: List[int]) -> List[int]:
        """Apply all learned BPE merges to *ids* in rank order."""
        while len(ids) >= 2:
            stats = get_stats(ids)
            best_pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if best_pair not in self.merges:
                break
            ids = merge(ids, best_pair, self.merges[best_pair])
        return ids

    # ------------------------------------------------------------------
    # Persistence (extend base)
    # ------------------------------------------------------------------

    def save(self, file_prefix: str) -> None:
        """Save tokenizer to ``<file_prefix>.model`` + ``.vocab``."""
        model_path = file_prefix + ".model"
        vocab_path = file_prefix + ".vocab"

        import json
        from .base import render_token

        model_data = {
            "pattern": self._pattern_str,
            "merges": [[list(pair), idx] for pair, idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)

        inverted_special = {v: k for k, v in self.special_tokens.items()}
        with open(vocab_path, "w", encoding="utf-8") as f:
            for idx, token_bytes in sorted(self.vocab.items()):
                if idx in inverted_special:
                    token_str = inverted_special[idx]
                else:
                    token_str = render_token(token_bytes)
                f.write(f"[{idx:6d}] {token_str}\n")

        print(f"Saved: {model_path}, {vocab_path}")

    def load(self, model_path: str) -> None:
        """Restore from a ``.model`` file produced by :meth:`save`."""
        import json

        assert model_path.endswith(".model")
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._pattern_str = data.get("pattern", GPT4_SPLIT_PATTERN)
        self._pattern = _compile(self._pattern_str)
        self.merges = {(p[0], p[1]): idx for p, idx in data["merges"]}
        self.special_tokens = data.get("special_tokens", {})
        self._build_vocab_from_merges()
        print(f"Loaded: {model_path}  vocab_size={self.vocab_size}")
