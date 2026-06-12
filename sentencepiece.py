"""
sentencepiece.py — SentencePiece-style BPE tokenizer

SentencePiece (Kudo & Richardson 2018) is the tokenizer used by LLaMA 1/2,
Mistral v1/v2, T5, mT5, ALBERT, and XLNet.  Unlike GPT-style BPE, it:

  1. Works entirely on the **Unicode code-point** level, not raw bytes,
     and uses a dedicated ▁ (U+2581, "Lower One Eighth Block") to mark
     the beginning of a word / space boundary.
  2. Is trained with a *unigram language model* by default (though BPE
     mode is also supported via ``--model_type=bpe``).
  3. Treats the whole input as a single stream — no pre-splitting regex.

This module provides:

  ``SentencePieceBPETokenizer``
      A pure-Python BPE tokenizer that replicates the SentencePiece **space
      normalization** convention (spaces → ▁), so tokens are compatible with
      vocabularies produced by Google's ``sentencepiece`` library.

  ``SentencePieceWrapper``
      A thin wrapper around the official ``sentencepiece`` Python package
      when it is installed.  Exposes the same ``encode`` / ``decode``
      interface as every other tokenizer in this project.

References
----------
  • Kudo & Richardson (2018) — https://arxiv.org/abs/1808.06226
  • SentencePiece GitHub — https://github.com/google/sentencepiece
  • LLaMA tokenizer — uses SentencePiece BPE with vocab_size=32000
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Tuple

from .base import Tokenizer, get_stats, merge

# The SentencePiece word-boundary marker (replaces leading spaces)
SPIECE_UNDERLINE = "▁"  # U+2581


class SentencePieceBPETokenizer(Tokenizer):
    """BPE tokenizer with SentencePiece-style space normalisation.

    This implementation replicates the space handling of SentencePiece's
    BPE mode:

    * Before tokenisation, all spaces in the input are replaced with ``▁``.
    * The ``▁`` character is treated as part of the token (unlike GPT-2's
      ``Ġ`` which is prepended to the *following* word).
    * Vocabulary IDs 0–255 are seeded from single characters (code points,
      not raw bytes) of the training corpus.

    Limitations compared to the full SentencePiece library:

    * No unigram LM training — only BPE.
    * No built-in sentence-boundary handling.
    * No byte-fallback for unseen characters (OOV raises ValueError).

    Parameters
    ----------
    add_bos:
        Prepend a ``<s>`` (beginning-of-sequence) token during encoding.
    add_eos:
        Append a ``</s>`` (end-of-sequence) token during encoding.
    """

    # Standard SentencePiece special tokens (IDs match LLaMA convention)
    SPECIAL_TOKENS: Dict[str, int] = {
        "<unk>": 0,
        "<s>":   1,
        "</s>":  2,
    }

    def __init__(self, add_bos: bool = False, add_eos: bool = False) -> None:
        # Don't call super().__init__() — we build a character vocab, not bytes
        self.merges: Dict[Tuple[int, int], int] = {}
        self.special_tokens: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.vocab: Dict[int, str] = {}          # id → string (not bytes)
        self.inverse_vocab: Dict[str, int] = {}  # string → id
        self.add_bos = add_bos
        self.add_eos = add_eos

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train BPE on *text* with SentencePiece space normalisation.

        Parameters
        ----------
        text:
            Training corpus.
        vocab_size:
            Target size (must be > number of unique characters + 3 specials).
        verbose:
            Print progress.
        """
        # Normalise spaces → ▁
        normalised = text.replace(" ", SPIECE_UNDERLINE)

        # Seed vocabulary: special tokens first, then unique characters
        self.vocab = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self.inverse_vocab = dict(self.SPECIAL_TOKENS)
        next_id = len(self.vocab)

        for char in sorted(set(normalised)):
            if char not in self.inverse_vocab:
                self.vocab[next_id] = char
                self.inverse_vocab[char] = next_id
                next_id += 1

        # Tokenise as character IDs
        ids: List[int] = []
        for char in normalised:
            if char in self.inverse_vocab:
                ids.append(self.inverse_vocab[char])
            else:
                ids.append(self.inverse_vocab["<unk>"])

        num_merges = vocab_size - len(self.vocab)
        if num_merges <= 0:
            return

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            new_id = next_id + i
            merged_str = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            ids = merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = merged_str
            self.inverse_vocab[merged_str] = new_id

            if verbose:
                print(
                    f"  merge {i+1:4d}/{num_merges}  "
                    f"({best_pair[0]}, {best_pair[1]}) → {new_id}  "
                    f"'{merged_str}'  freq={stats[best_pair]}"
                )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode *text* to a list of integer token IDs.

        Applies SentencePiece space-normalisation (space → ▁) before
        tokenisation.  BOS/EOS tokens are added if requested at init.
        """
        normalised = text.replace(" ", SPIECE_UNDERLINE)

        # Character-level seed IDs
        ids: List[int] = []
        for char in normalised:
            if char in self.inverse_vocab:
                ids.append(self.inverse_vocab[char])
            else:
                ids.append(self.inverse_vocab.get("<unk>", 0))

        # Apply merges in rank order
        while len(ids) >= 2:
            stats = get_stats(ids)
            best_pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if best_pair not in self.merges:
                break
            ids = merge(ids, best_pair, self.merges[best_pair])

        if self.add_bos:
            ids = [self.special_tokens["<s>"]] + ids
        if self.add_eos:
            ids = ids + [self.special_tokens["</s>"]]

        return ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to a string.

        Strips the leading BOS / EOS if present, then reverses the
        space-normalisation (▁ → space, with the leading ▁ dropped).
        """
        tokens = []
        for i in ids:
            tok = self.vocab.get(i, "<unk>")
            # Skip BOS / EOS control tokens in output
            if tok in ("<s>", "</s>"):
                continue
            tokens.append(tok)

        text = "".join(tokens)
        # ▁ at the start of a token = word boundary / space
        # The very first ▁ (start of sequence) is dropped; rest become spaces
        text = text.replace(SPIECE_UNDERLINE, " ").strip()
        return text

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return (
            f"SentencePieceBPETokenizer("
            f"vocab_size={self.vocab_size}, "
            f"merges={len(self.merges)}, "
            f"add_bos={self.add_bos}, add_eos={self.add_eos})"
        )


# ---------------------------------------------------------------------------
# Optional: thin wrapper around the official sentencepiece package
# ---------------------------------------------------------------------------

class SentencePieceWrapper:
    """Wraps the ``sentencepiece`` Python package with the project's interface.

    Install via:  ``pip install sentencepiece``

    Use this class for production inference with LLaMA 1/2, Mistral v1/v2, or
    any other model that ships a ``.model`` file produced by the C++ library.

    Parameters
    ----------
    model_path:
        Path to a ``*.model`` file (e.g. ``tokenizer.model`` from Meta's
        LLaMA release).
    add_bos:
        Prepend BOS token.
    add_eos:
        Append EOS token.
    """

    def __init__(
        self,
        model_path: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> None:
        try:
            import sentencepiece as spm  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'sentencepiece' package is required for SentencePieceWrapper.\n"
                "Install it with:  pip install sentencepiece"
            ) from exc

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)
        self.add_bos = add_bos
        self.add_eos = add_eos

    @property
    def vocab_size(self) -> int:
        return self._sp.GetPieceSize()

    def encode(self, text: str) -> List[int]:
        """Encode *text* using the loaded SentencePiece model."""
        return self._sp.Encode(text, add_bos=self.add_bos, add_eos=self.add_eos)

    def decode(self, ids: List[int]) -> str:
        """Decode *ids* back to a string."""
        return self._sp.Decode(ids)

    def token_to_id(self, token: str) -> int:
        return self._sp.PieceToId(token)

    def id_to_token(self, idx: int) -> str:
        return self._sp.IdToPiece(idx)

    def __repr__(self) -> str:
        return f"SentencePieceWrapper(vocab_size={self.vocab_size})"
