"""
train.py — Train a tokenizer from a text corpus and save it to disk.

Usage examples
--------------

# Train a BasicTokenizer with vocab_size=1000
python train.py --input corpus.txt --vocab_size 1000 --type basic --output models/basic

# Train a RegexTokenizer (GPT-4 pattern) with verbose output
python train.py --input corpus.txt --vocab_size 4096 --type regex --pattern gpt4 --output models/regex --verbose

# Train a SentencePiece-style tokenizer
python train.py --input corpus.txt --vocab_size 32000 --type sentencepiece --output models/sp
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Allow running from project root without installing
sys.path.insert(0, os.path.dirname(__file__))

from tokenizers import (
    BasicTokenizer,
    RegexTokenizer,
    SentencePieceBPETokenizer,
    GPT2_SPLIT_PATTERN,
    GPT4_SPLIT_PATTERN,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a BPE tokenizer and save to disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      required=True,  help="Path to training text file (UTF-8).")
    p.add_argument("--output",     required=True,  help="Output file prefix (no extension).")
    p.add_argument("--vocab_size", type=int, default=512, help="Target vocabulary size.")
    p.add_argument(
        "--type",
        choices=["basic", "regex", "sentencepiece"],
        default="regex",
        help="Tokenizer variant to train.",
    )
    p.add_argument(
        "--pattern",
        choices=["gpt2", "gpt4"],
        default="gpt4",
        help="Regex split pattern (only for --type regex).",
    )
    p.add_argument("--verbose", action="store_true", help="Print merge progress.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Load training corpus
    # ------------------------------------------------------------------ #
    print(f"Loading corpus: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  {len(text):,} characters loaded.")

    # ------------------------------------------------------------------ #
    # Build tokenizer
    # ------------------------------------------------------------------ #
    if args.type == "basic":
        tokenizer = BasicTokenizer()
    elif args.type == "regex":
        pattern = GPT2_SPLIT_PATTERN if args.pattern == "gpt2" else GPT4_SPLIT_PATTERN
        tokenizer = RegexTokenizer(pattern=pattern)
    elif args.type == "sentencepiece":
        tokenizer = SentencePieceBPETokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {args.type}")

    print(f"\nTraining {tokenizer.__class__.__name__} | vocab_size={args.vocab_size}")
    t0 = time.time()
    tokenizer.train(text, vocab_size=args.vocab_size, verbose=args.verbose)
    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.2f}s.")

    # ------------------------------------------------------------------ #
    # Quick sanity check
    # ------------------------------------------------------------------ #
    sample = text[:200]
    encoded = tokenizer.encode(sample) if args.type != "sentencepiece" else tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"\nSanity check (first 200 chars):")
    print(f"  original : {sample!r}")
    print(f"  encoded  : {encoded[:20]}{'...' if len(encoded)>20 else ''}")
    print(f"  decoded  : {decoded[:200]!r}")
    match = "✅ match" if decoded == sample else "⚠️  MISMATCH"
    print(f"  result   : {match}")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tokenizer.save(args.output)
    print(f"\nSaved to: {args.output}.model  +  {args.output}.vocab")
    print(f"Final vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
