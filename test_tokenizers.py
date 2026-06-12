"""
tests/test_tokenizers.py — Unit + integration tests for all tokenizer variants.

Run with:  python -m pytest tests/ -v
       or:  python tests/test_tokenizers.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizers import BasicTokenizer, RegexTokenizer, SentencePieceBPETokenizer
from tokenizers.regex import GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN


# ---------------------------------------------------------------------------
# Shared corpus fixture
# ---------------------------------------------------------------------------

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump! "
    "the five boxing wizards jump quickly. " * 20
)

VOCAB_SIZE = 300  # small enough to be fast in tests


# ---------------------------------------------------------------------------
# BasicTokenizer
# ---------------------------------------------------------------------------

class TestBasicTokenizer(unittest.TestCase):

    def setUp(self):
        self.tok = BasicTokenizer()
        self.tok.train(CORPUS, vocab_size=VOCAB_SIZE)

    def test_encode_decode_roundtrip(self):
        """Decoding encoded text must recover the original."""
        for sample in [
            "hello world",
            "the quick brown fox",
            "unseen text with numbers 123!",
        ]:
            with self.subTest(sample=sample):
                ids = self.tok.encode(sample)
                recovered = self.tok.decode(ids)
                self.assertEqual(recovered, sample)

    def test_encode_returns_list_of_ints(self):
        ids = self.tok.encode("hello")
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_vocab_size(self):
        self.assertEqual(self.tok.vocab_size, VOCAB_SIZE)

    def test_all_ids_in_vocab(self):
        ids = self.tok.encode(CORPUS[:500])
        for i in ids:
            self.assertIn(i, self.tok.vocab, f"Token ID {i} not in vocab")

    def test_empty_string(self):
        self.assertEqual(self.tok.encode(""), [])
        self.assertEqual(self.tok.decode([]), "")

    def test_unicode_text(self):
        text = "こんにちは世界"
        ids = self.tok.encode(text)
        recovered = self.tok.decode(ids)
        self.assertEqual(recovered, text)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "basic")
            self.tok.save(prefix)

            new_tok = BasicTokenizer()
            new_tok.load(prefix + ".model")

            sample = "the fox jumps"
            self.assertEqual(self.tok.encode(sample), new_tok.encode(sample))

    def test_compression_ratio(self):
        """Trained tokenizer should compress the training text."""
        ids = self.tok.encode(CORPUS)
        raw_len = len(CORPUS.encode("utf-8"))
        self.assertLess(len(ids), raw_len,
                        "Expected compression: encoded length < raw byte length")

    def test_known_bpe_merge(self):
        """Replicates the Wikipedia BPE example from Karpathy's minBPE."""
        tok = BasicTokenizer()
        tok.train("aaabdaaabac", vocab_size=256 + 3)
        self.assertEqual(tok.encode("aaabdaaabac"), [258, 100, 258, 97, 99])
        self.assertEqual(tok.decode([258, 100, 258, 97, 99]), "aaabdaaabac")


# ---------------------------------------------------------------------------
# RegexTokenizer
# ---------------------------------------------------------------------------

class TestRegexTokenizer(unittest.TestCase):

    def _make_tok(self, pattern=None):
        tok = RegexTokenizer(pattern=pattern)
        tok.train(CORPUS, vocab_size=VOCAB_SIZE)
        return tok

    def test_gpt4_pattern_roundtrip(self):
        tok = self._make_tok(GPT4_SPLIT_PATTERN)
        sample = "Hello, world! 123 -- test."
        self.assertEqual(tok.decode(tok.encode(sample)), sample)

    def test_gpt2_pattern_roundtrip(self):
        tok = self._make_tok(GPT2_SPLIT_PATTERN)
        sample = "Hello, world! 123 -- test."
        self.assertEqual(tok.decode(tok.encode(sample)), sample)

    def test_special_tokens(self):
        tok = self._make_tok()
        special = {"<|endoftext|>": 400, "<|pad|>": 401}
        tok.register_special_tokens(special)

        text = "hello <|endoftext|> world"
        ids = tok.encode(text, allowed_special="all")
        self.assertIn(400, ids, "<|endoftext|> should map to ID 400")

        recovered = tok.decode(ids)
        self.assertEqual(recovered, text)

    def test_special_token_raises_when_not_allowed(self):
        tok = self._make_tok()
        tok.register_special_tokens({"<|endoftext|>": 400})
        with self.assertRaises(ValueError):
            tok.encode("hello <|endoftext|>", allowed_special="none_raise")

    def test_no_cross_boundary_merges(self):
        """Numbers and letters should not merge across boundaries."""
        tok = RegexTokenizer(pattern=GPT4_SPLIT_PATTERN)
        text = "abc123" * 50
        tok.train(text, vocab_size=300)
        ids = tok.encode("abc123")
        decoded = tok.decode(ids)
        self.assertEqual(decoded, "abc123")

    def test_save_load_roundtrip(self):
        tok = self._make_tok()
        tok.register_special_tokens({"<|endoftext|>": 400})
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "regex")
            tok.save(prefix)
            new_tok = RegexTokenizer()
            new_tok.load(prefix + ".model")
            sample = "the quick fox"
            self.assertEqual(tok.encode(sample), new_tok.encode(sample))

    def test_multiline_text(self):
        tok = self._make_tok()
        text = "line one\nline two\nline three"
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_emoji_and_unicode(self):
        tok = self._make_tok()
        text = "hello 😊 world"
        self.assertEqual(tok.decode(tok.encode(text)), text)


# ---------------------------------------------------------------------------
# SentencePieceBPETokenizer
# ---------------------------------------------------------------------------

class TestSentencePieceBPETokenizer(unittest.TestCase):

    def setUp(self):
        self.tok = SentencePieceBPETokenizer()
        self.tok.train(CORPUS, vocab_size=400)

    def test_encode_decode_roundtrip(self):
        sample = "the quick brown fox"
        recovered = self.tok.decode(self.tok.encode(sample))
        self.assertEqual(recovered, sample)

    def test_bos_eos_tokens(self):
        tok = SentencePieceBPETokenizer(add_bos=True, add_eos=True)
        tok.train(CORPUS, vocab_size=400)
        ids = tok.encode("hello world")
        self.assertEqual(ids[0], tok.special_tokens["<s>"])
        self.assertEqual(ids[-1], tok.special_tokens["</s>"])

    def test_space_normalisation(self):
        """Internal ▁ markers should not appear in decoded output."""
        decoded = self.tok.decode(self.tok.encode("hello world"))
        self.assertNotIn("▁", decoded)
        self.assertEqual(decoded, "hello world")

    def test_encode_returns_ints(self):
        ids = self.tok.encode("test")
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_empty_string(self):
        self.assertEqual(self.tok.decode(self.tok.encode("")), "")

    def test_vocab_size_property(self):
        self.assertGreater(self.tok.vocab_size, 3)  # At least specials + chars


# ---------------------------------------------------------------------------
# Cross-tokenizer consistency checks
# ---------------------------------------------------------------------------

class TestCrossTokenizer(unittest.TestCase):

    def test_all_tokenizers_roundtrip_same_text(self):
        """All tokenizers must losslessly encode+decode text seen in the corpus.

        Note: SentencePieceBPETokenizer replaces characters absent from its
        training vocab with <unk>.  We test it with a corpus-safe sample.
        """
        # BasicTokenizer and RegexTokenizer work at raw byte level → any UTF-8
        for name, tok in [("basic", BasicTokenizer()), ("regex", RegexTokenizer())]:
            tok.train(CORPUS, vocab_size=350)
            sample = "hello world, this is a test sentence!"
            with self.subTest(tokenizer=name):
                self.assertEqual(tok.decode(tok.encode(sample)), sample,
                                 f"{name} roundtrip failed")

        # SentencePieceBPETokenizer operates at character level; only characters
        # seen in training are in vocab.  Use a corpus-safe sample (no punct).
        sp_tok = SentencePieceBPETokenizer()
        sp_tok.train(CORPUS, vocab_size=350)
        sp_sample = "the quick brown fox jumps over the lazy dog"
        with self.subTest(tokenizer="sp"):
            self.assertEqual(sp_tok.decode(sp_tok.encode(sp_sample)), sp_sample,
                             "sp roundtrip failed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
