"""
basic.py — BasicTokenizer

The simplest byte-level BPE tokenizer.  Operates directly on the raw
UTF-8 bytes of the input text — no regex pre-splitting, no special token
machinery.  Great for understanding the algorithm end-to-end.

Influenced by:
  • Karpathy's minBPE  (https://github.com/karpathy/minbpe)
  • Raschka's BPE from scratch (https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)

Quick start
-----------
>>> from tokenizers.basic import BasicTokenizer
>>> tok = BasicTokenizer()
>>> tok.train("aaabdaaabac", vocab_size=256 + 3)
>>> tok.encode("aaabdaaabac")
[258, 100, 258, 97, 99]
>>> tok.decode([258, 100, 258, 97, 99])
'aaabdaaabac'
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    """Byte-level BPE without regex pre-splitting.

    The full pipeline:

    1. Encode input text to raw UTF-8 bytes → initial token IDs (0-255).
    2. Repeatedly find the most frequent adjacent pair.
    3. Replace every occurrence of that pair with a new token ID (256+).
    4. Record the merge rule.
    5. Stop when ``vocab_size`` is reached or no more pairs exist.

    Encoding (inference) replays the *same* merge rules in rank order.
    Decoding maps token IDs back to bytes and converts to UTF-8.
    """

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn BPE merges from *text*.

        Parameters
        ----------
        text:
            Training corpus (plain Python str).
        vocab_size:
            Target vocabulary size.  Must be >= 256 (the initial byte vocab).
        verbose:
            If True, print a progress line for each merge.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256"
        num_merges = vocab_size - 256

        # Encode the training text to bytes, cast to list of ints
        ids: List[int] = list(text.encode("utf-8"))
        original_len = len(ids)

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        t0 = time.time()
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                if verbose:
                    print(f"[Merge {i+1}/{num_merges}] No pairs remain — stopping early.")
                break

            best_pair = max(stats, key=stats.get)
            new_id = 256 + i

            ids = merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose:
                elapsed = time.time() - t0
                token_str = self.vocab[new_id].decode("utf-8", errors="replace")
                print(
                    f"  merge {i+1:4d}/{num_merges} "
                    f"({best_pair[0]}, {best_pair[1]}) → {new_id}  "
                    f"'{token_str}'  "
                    f"freq={stats[best_pair]}  "
                    f"seq_len={len(ids)}  "
                    f"elapsed={elapsed:.1f}s"
                )

        ratio = original_len / len(ids) if ids else 1.0
        if verbose:
            print(
                f"\nTraining complete: {num_merges} merges, "
                f"vocab_size={self.vocab_size}, "
                f"compression ratio={ratio:.2f}x "
                f"({original_len} → {len(ids)} tokens)"
            )

    def encode(self, text: str) -> List[int]:
        """Encode *text* to a list of token IDs.

        Parameters
        ----------
        text:
            Input string.

        Returns
        -------
        List of integer token IDs.
        """
        ids: List[int] = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = get_stats(ids)
            # Pick the pair with the *lowest* merge rank (i.e. learned earliest)
            best_pair = min(
                stats,
                key=lambda p: self.merges.get(p, float("inf")),
            )
            if best_pair not in self.merges:
                break  # No more merges applicable
            ids = merge(ids, best_pair, self.merges[best_pair])

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to a string.

        Parameters
        ----------
        ids:
            List produced by :meth:`encode`.

        Returns
        -------
        Decoded UTF-8 string.  Undecodable bytes are replaced with the
        Unicode replacement character (U+FFFD).
        """
        token_bytes = b"".join(self.vocab[i] for i in ids)
        return token_bytes.decode("utf-8", errors="replace")
