# 🔤 LLM Tokenization — The Complete Reference

> **One-stop guide** covering everything from raw bytes to modern tokenizers used in GPT-2, GPT-4, LLaMA, Mistral, Falcon, and beyond. Includes theory, math, production code walkthroughs, and decision guides.

---

## Table of Contents

1. [What is Tokenization and Why Does it Matter?](#1-what-is-tokenization-and-why-does-it-matter)
2. [A Brief History of Tokenization](#2-a-brief-history-of-tokenization)
3. [Bits, Bytes, and Unicode — The Foundation](#3-bits-bytes-and-unicode--the-foundation)
4. [Tokenization Strategies: From Simple to Modern](#4-tokenization-strategies-from-simple-to-modern)
5. [Byte Pair Encoding (BPE) — Deep Dive](#5-byte-pair-encoding-bpe--deep-dive)
6. [SentencePiece — Deep Dive](#6-sentencepiece--deep-dive)
7. [BPE vs SentencePiece — Head-to-Head](#7-bpe-vs-sentencepiece--head-to-head)
8. [Tokenizers Used by Modern LLMs](#8-tokenizers-used-by-modern-llms)
9. [Code Walkthrough — This Repository](#9-code-walkthrough--this-repository)
10. [Special Tokens Demystified](#10-special-tokens-demystified)
11. [Vocabulary Size: How to Choose](#11-vocabulary-size-how-to-choose)
12. [Tokenizer Pathologies and Edge Cases](#12-tokenizer-pathologies-and-edge-cases)
13. [Multilingual Tokenization](#13-multilingual-tokenization)
14. [Training Your Own Tokenizer](#14-training-your-own-tokenizer)
15. [Quick Reference: Model-to-Tokenizer Map](#15-quick-reference-model-to-tokenizer-map)
16. [Key Papers and Resources](#16-key-papers-and-resources)
17. [Deep Concepts: Tokenization and Model Cognition](#17-deep-concepts-tokenization-and-model-cognition)
18. [WordPiece and Unigram LM — Full Derivations](#18-wordpiece-and-unigram-lm--full-derivations)
19. [Information-Theoretic View of Tokenization](#19-information-theoretic-view-of-tokenization)
20. [Tokenization as a Lossy Compression Problem](#20-tokenization-as-a-lossy-compression-problem)
21. [Tokenizer-Model Co-design: The Hidden Coupling](#21-tokenizer-model-co-design-the-hidden-coupling)
22. [Prompt Engineering at the Token Level](#22-prompt-engineering-at-the-token-level)
23. [Tokenization in Multimodal Models](#23-tokenization-in-multimodal-models)
24. [Adversarial Tokenization and Security](#24-adversarial-tokenization-and-security)
25. [Future Directions: Tokenizer-Free and Dynamic Tokenization](#25-future-directions-tokenizer-free-and-dynamic-tokenization)

---

## 1. What is Tokenization and Why Does it Matter?

A Large Language Model (LLM) cannot read text. It reads **integers**. Tokenization is the process of mapping a raw string into a sequence of integers (token IDs) that can be fed into an embedding layer.

```
"Hello, world!"  →  [15496, 11, 995, 0]   (GPT-2)
                 →  [9906, 11, 1917, 0]    (GPT-4 / cl100k_base)
                 →  [1, 15043, 29892, 3186, 29991]  (LLaMA 2)
```

**Why the tokenizer is not "just preprocessing":**

| Design choice | Downstream effect |
|---|---|
| Vocabulary size | Controls model size (embedding table rows × embedding dim) |
| Token boundaries | Determines what the model sees as a "unit of meaning" |
| Space handling | Governs how words at sentence boundaries look to the model |
| Number tokenisation | Affects arithmetic reasoning (GPT-2 splits `1234` into `12`, `34`) |
| Special tokens | Enable instruction following, chat turns, tool calls |

A poor tokeniser can make simple arithmetic hard, break multilingual performance, or cause invisible prompt-injection vulnerabilities.

---

## 2. A Brief History of Tokenization

```
1950s–1990s  │  Whitespace / punctuation splitting ("Moses" tokenizer)
             │  Rule-based: split on spaces, strip punctuation
             │
1994         │  Philip Gage publishes BPE for data compression
             │  (Originally for compressing source code)
             │
2016         │  Sennrich et al. adapt BPE for Neural Machine Translation
             │  "Neural Machine Translation of Rare Words with Subword Units"
             │
2018         │  Google's SentencePiece released (Kudo & Richardson)
             │  Unigram LM + BPE, language-agnostic, no pre-tokenisation needed
             │
2019         │  GPT-2 ships byte-level BPE with regex pre-splitting
             │  OpenAI open-sources encoder.py — influences all future LLMs
             │
2020         │  GPT-3 uses a larger BPE vocabulary (50,257 → effectively same)
             │
2021-23      │  LLaMA 1/2 use SentencePiece BPE (vocab = 32,000)
             │  Falcon, Mistral v1/v2 also use SentencePiece
             │
2023         │  GPT-4 / cl100k_base: improved regex, 100,256 vocab
             │  LLaMA 3: switches to tiktoken (cl100k_base-style), 128,000 vocab
             │  Mistral v3+: tiktoken, 131,072 vocab
             │
2024–present │  Gemma, Phi-3, Qwen: diverse vocabularies (32K–150K)
             │  Trend: larger vocabularies, tiktoken dominance
```

---

## 3. Bits, Bytes, and Unicode — The Foundation

### 3.1 From Characters to Bytes

Everything in a computer is ultimately bits (0s and 1s). A **byte** is 8 bits, giving 256 possible values (0–255). When you write text in Python and call `.encode("utf-8")`, you get a sequence of these byte values.

```python
text = "Hello"
byte_list = list(text.encode("utf-8"))
# → [72, 101, 108, 108, 111]
```

This gives us an immediately usable integer representation — but it's inefficient. "Hello" becomes 5 tokens, each representing exactly one character. A 1000-character document = 1000 tokens. Every LLM has a **context window** limit (e.g. 4096, 8192, 128K tokens), so wasteful tokenisation directly costs you context.

### 3.2 Unicode

Unicode assigns a unique **code point** to every character in every human writing system. There are over 140,000 code points.

**UTF-8** encodes these code points into 1–4 bytes:

| Characters | Bytes used |
|---|---|
| ASCII (A–Z, 0–9, punctuation) | 1 byte |
| Latin Extended, Greek, Cyrillic | 2 bytes |
| Chinese, Japanese, Korean (CJK) | 3 bytes |
| Emoji, rare scripts | 4 bytes |

This means a single Chinese character becomes **3 tokens** under a naive byte-level approach, while the same "semantic unit" as an English word might be 1 token. This is why multilingual models benefit enormously from proper BPE/SentencePiece training.

### 3.3 Why 256 as the Starting Point

Since a byte has 256 possible values, byte-level BPE tokenizers start their vocabulary with all 256 byte values (IDs 0–255). This is a key design choice: it means the tokenizer can represent **any valid UTF-8 string** — there are no out-of-vocabulary (OOV) tokens at the byte level. Unlike word-level tokenizers that fail on unseen words, byte-level BPE gracefully handles any input by falling back to individual bytes.

---

## 4. Tokenization Strategies: From Simple to Modern

### 4.1 Word-level Tokenization

Split on whitespace. Map each word to an ID.

```
"the cat sat" → ["the", "cat", "sat"] → [1, 2, 3]
```

**Problems:**
- "cat" and "cats" are different tokens — the model must learn their relationship from scratch
- "unrecognised" → `<UNK>` (catastrophic for rare technical terms)
- Vocabulary can easily exceed 1 million words in multilingual settings

Used by: Word2Vec, early LSTM language models. Not used in modern LLMs.

### 4.2 Character-level Tokenization

Every character is one token.

```
"hello" → ["h", "e", "l", "l", "o"] → [1, 2, 3, 3, 4]
```

**Problems:**
- Extremely long sequences — a 500-word essay = ~2500 tokens
- The model must learn spelling, morphology, and semantics all from scratch
- Computationally very expensive due to long sequences

Explored in research (e.g. ByT5), but not used in production LLMs.

### 4.3 Subword Tokenization (Modern Standard)

The sweet spot: merge frequent character sequences into subword units.

```
"tokenization" → ["token", "ization"]       (BPE)
"tokenization" → ["▁token", "ization"]      (SentencePiece)
"tokenization" → ["token", "##ization"]     (WordPiece / BERT)
```

**Properties:**
- Common words appear as single tokens
- Rare words are split into meaningful subwords
- Any string can be encoded (no OOV at byte level)
- Fixed, bounded vocabulary size

---

## 5. Byte Pair Encoding (BPE) — Deep Dive

### 5.1 The Original Algorithm (Gage, 1994)

BPE was invented for **data compression**, not NLP. The idea is simple:

> Find the most frequent adjacent pair of symbols in your data. Replace every occurrence with a new single symbol. Repeat.

**Worked example — encoding `the cat in the hat`:**

```
Initial:  t h e   c a t   i n   t h e   h a t
          (each character is its own symbol)

Step 1 — most frequent pair = (t, h) → merge to 'th'
          th e   c a t   i n   th e   h a t
          New vocab entry: 256 → "th"

Step 2 — most frequent pair = (th, e) → merge to 'the'
          the   c a t   i n   the   h a t
          New vocab entry: 257 → "the"

Step 3 — most frequent pair = (the, ' ') → merge to 'the '
          the c a t   i n   the h a t
          New vocab entry: 258 → "the "
...
```

### 5.2 Applying BPE to Language (Sennrich et al., 2016)

The NMT adaptation does the same thing but operates on **word-level pre-tokenised** text. The training corpus is first split by whitespace, then BPE merges are learned over character sequences **within** words. This prevents merges from crossing word boundaries.

### 5.3 GPT-2 Style: Byte-Level BPE with Regex Pre-Splitting

OpenAI made two critical changes for GPT-2:

**Change 1: Operate on raw UTF-8 bytes, not Unicode characters**

This eliminates OOV entirely. The vocabulary seeds with all 256 possible bytes.

**Change 2: Pre-split text with a regex before running BPE**

```python
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
```

This pattern splits the text into linguistic chunks before BPE: contractions (`'re`, `'ll`), words (letters optionally preceded by a space), numbers (max 3 digits), punctuation/symbols, and whitespace. BPE merges are then applied **independently within each chunk**.

**Why this matters:** Without pre-splitting, BPE might learn to merge `e` + ` ` + `t` (from `the` + ` ` + `the`) into a token that spans a word boundary. Pre-splitting prevents this, leading to more linguistically sensible tokens.

**Space encoding — the `Ġ` trick (GPT-2):**

Rather than treating spaces as separate tokens, GPT-2 prepends a special `Ġ` (U+0120) character to words that follow a space:

```
"Hello world" → ["Hello", "Ġworld"]
```

This means the model always knows, from a single token, whether a word is at the start of a sentence or mid-sentence. The GPT-4 tokenizer improves this by simply including the space as a literal byte in the token (` world` instead of `Ġworld`).

### 5.4 The BPE Training Algorithm — Step by Step

```
INPUT:  text corpus,  target vocab_size V
OUTPUT: merge table  M = {(a, b) → new_id}, vocabulary {id → bytes}

1. Encode corpus as raw UTF-8 bytes
   ids = list(corpus.encode("utf-8"))
   # ids is now a list of integers, each 0–255

2. Initialise vocabulary
   vocab = {i: bytes([i]) for i in range(256)}
   # 256 single-byte tokens

3. Repeat until len(vocab) == V:
   a. Count all adjacent pairs
      stats = Counter(zip(ids, ids[1:]))

   b. Find the most frequent pair
      best = max(stats, key=stats.get)

   c. Assign new ID
      new_id = len(vocab)

   d. Merge: replace every (best[0], best[1]) in ids with new_id
      ids = merge(ids, best, new_id)

   e. Record the merge
      M[best] = new_id
      vocab[new_id] = vocab[best[0]] + vocab[best[1]]

4. Return M, vocab
```

**Time complexity:** O(N × V) where N = corpus length, V = merges to perform. This is why large tokenizer training can take hours on huge corpora.

### 5.5 BPE Encoding (Inference)

Given a trained merge table, encoding a new string:

```
1. Encode string as UTF-8 bytes → list of IDs (0–255)
2. While len(ids) >= 2:
     stats = {pair: count for pair in zip(ids, ids[1:])}
     best = min(stats, key=lambda p: merge_rank.get(p, ∞))
     if best not in merges: break
     ids = merge(ids, best, merges[best])
3. Return ids
```

The key insight: apply merges in **rank order** (the order they were learned during training). Earlier merges were for more frequent pairs and should be applied first.

### 5.6 BPE Decoding

```
1. For each id in token_ids:
     look up vocab[id] → bytes
2. Concatenate all bytes
3. Decode as UTF-8 (with error handling)
```

Decoding is O(N) and always reversible — a crucial property.

---

## 6. SentencePiece — Deep Dive

### 6.1 Motivation

Google developed SentencePiece (2018) to solve a key limitation of GPT-2-style BPE: **it requires pre-tokenised text**. Languages like Chinese, Japanese, and Thai have no spaces between words, making whitespace-based pre-splitting impossible.

SentencePiece treats the input as a **raw stream of Unicode characters** with no language-specific preprocessing. It defines its own notion of word boundary using the `▁` (U+2581) marker.

### 6.2 The `▁` (Underline) Convention

SentencePiece replaces every space in the input with `▁` and then tokenises the entire string as a continuous character sequence:

```
Input:   "Hello world"
Internal: "Hello▁world"
Tokens:  ["▁Hello", "▁world"]     ← each token includes its leading ▁
          OR
         ["Hello", "▁world"]      ← first word has no preceding space
```

During decoding, every `▁` is replaced with a space, and the result is trimmed. This encodes positional information (word-initial vs. word-internal) directly into each token.

### 6.3 Two Training Algorithms in SentencePiece

**BPE mode** (`--model_type=bpe`):
Runs the standard BPE algorithm on the character-level stream. This is what LLaMA 1/2 and early Mistral use.

**Unigram Language Model mode** (`--model_type=unigram`)  — **the default**:

Instead of bottom-up merging, Unigram LM starts with a large candidate vocabulary and **prunes** it:

```
1. Start with a large seed vocabulary (all substrings up to length L)
2. Use the EM algorithm to fit a unigram LM:
     P(text) = ∏ P(token_i)    (assuming independent tokens)
3. Remove the X% of tokens whose removal causes the least increase in loss
4. Repeat until vocab_size is reached
```

Unigram LM produces a **probabilistic tokeniser**: a given text can have multiple valid tokenisations, and the algorithm picks the most probable one (Viterbi decoding). This is different from BPE, which is deterministic.

**SentencePiece noise (subword regularisation):** During training, SentencePiece can sample from the distribution of valid tokenisations (not just the most probable one). This acts as data augmentation and improves model robustness.

### 6.4 Special Tokens in SentencePiece

SentencePiece reserves the first few IDs by convention:

| ID | Token | Meaning |
|---|---|---|
| 0 | `<unk>` | Unknown / OOV character |
| 1 | `<s>` | Beginning of sequence (BOS) |
| 2 | `</s>` | End of sequence (EOS) |

Models like LLaMA use these IDs exactly. The BOS token is automatically prepended during encoding (configurable).

### 6.5 Byte Fallback

SentencePiece supports a `--byte_fallback` option. When enabled, any character not in the vocabulary is split into its UTF-8 bytes (represented as special tokens like `<0xE4>`). This gives it the same OOV-free guarantee as byte-level BPE.

LLaMA 2 enables byte fallback. LLaMA 3 switches to tiktoken (byte-level BPE) instead.

---

## 7. BPE vs SentencePiece — Head-to-Head

| Dimension | BPE (tiktoken / GPT-style) | SentencePiece |
|---|---|---|
| Unit of operation | Raw UTF-8 bytes | Unicode code points |
| Space handling | Included in token bytes (` world`) or `Ġworld` | `▁world` marker |
| Pre-splitting | Regex (GPT-2/4 pattern) | None — full stream |
| OOV handling | Impossible (byte-level fallback) | `<unk>` or byte fallback |
| Training algorithm | BPE (bottom-up merge) | BPE or Unigram LM |
| Probabilistic | No — deterministic | Yes (Unigram LM mode) |
| Subword regularisation | Not standard | Built-in |
| Library | tiktoken (fast C++), HuggingFace | sentencepiece (C++) |
| Languages | Works well; larger vocab for non-ASCII | Excellent; no pre-split needed |
| Used by | GPT-2/3/4, LLaMA 3, Mistral v3+ | LLaMA 1/2, Mistral v1/v2, T5, Falcon |

**When to choose BPE (tiktoken-style):**
- Building a GPT-style model for primarily English text
- You want exact compatibility with OpenAI's tokenizer
- You prefer a deterministic, byte-safe tokenizer
- You're fine with the `regex` dependency

**When to choose SentencePiece:**
- Multilingual model (especially CJK languages)
- You want subword regularisation for training robustness
- You need BOS/EOS tokens built into the library
- Compatibility with LLaMA 1/2, T5, mT5, ALBERT

---

## 8. Tokenizers Used by Modern LLMs

### 8.1 GPT-2 (OpenAI, 2019)

| Property | Value |
|---|---|
| Algorithm | Byte-level BPE |
| Vocabulary size | 50,257 |
| Special tokens | `<\|endoftext\|>` (ID 50256) |
| Space encoding | `Ġ` prefix (U+0120) |
| Pre-split regex | GPT-2 pattern |
| Library | tiktoken (`gpt2` encoding) |

The initial 256 vocab tokens (IDs 0–255) map to individual bytes, but with a special display encoding. Tokens 256+ are learned merges. The vocabulary has exactly 50,256 merge tokens plus `<|endoftext|>`.

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
enc.encode("Hello, world!")   # → [15496, 11, 995, 0]
enc.decode([15496, 11, 995, 0])  # → "Hello, world!"
```

### 8.2 GPT-4 / cl100k_base (OpenAI, 2023)

| Property | Value |
|---|---|
| Algorithm | Byte-level BPE |
| Vocabulary size | 100,256 base + special tokens |
| Special tokens | `<\|endoftext\|>`, `<\|fim_prefix\|>`, `<\|fim_middle\|>`, `<\|fim_suffix\|>`, `<\|endofprompt\|>` |
| Space encoding | Literal space byte in token |
| Pre-split regex | GPT-4 pattern (improved from GPT-2) |
| Library | tiktoken (`cl100k_base` encoding) |

Key improvements over GPT-2:
- Double the vocabulary → tokens are on average ~2× longer (more efficient)
- Better number tokenisation: 3-digit numbers mostly single tokens
- Case-consistent tokenisation (GPT-2 sometimes splits differently based on capitalisation)
- Improved handling of whitespace and newlines

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
enc.encode("Hello, world!")   # → [9906, 11, 1917, 0]
```

### 8.3 LLaMA 1 and LLaMA 2 (Meta, 2023)

| Property | Value |
|---|---|
| Algorithm | SentencePiece BPE |
| Vocabulary size | 32,000 |
| Special tokens | `<unk>` (0), `<s>` (1), `</s>` (2) |
| Space encoding | `▁` marker (SentencePiece convention) |
| BOS / EOS | Auto-prepended by library |
| Byte fallback | Yes (`<0xNN>` tokens for OOV bytes) |
| Library | sentencepiece |

The 32K vocabulary was adequate for English but stretched thin for multilingual content. CJK characters each need multiple tokens, hurting efficiency for those languages.

```python
# Using the official sentencepiece model from Meta
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
sp.Encode("Hello, world!")   # → [1, 15043, 29892, 3186, 29991]
sp.Decode([1, 15043, 29892, 3186, 29991])  # → "<s> Hello, world!"
```

### 8.4 LLaMA 3 (Meta, 2024)

| Property | Value |
|---|---|
| Algorithm | Byte-level BPE (tiktoken-style) |
| Vocabulary size | 128,000 |
| Special tokens | 256 reserved slots (`<\|begin_of_text\|>`, `<\|end_of_text\|>`, `<\|eot_id\|>`, etc.) |
| Space encoding | Literal space (like GPT-4) |
| Pre-split regex | GPT-4 pattern |
| Library | tiktoken |

Meta's switch from SentencePiece to tiktoken for LLaMA 3 is significant. The 4× larger vocabulary (32K → 128K) dramatically improves tokenisation efficiency, especially for code and multilingual text. A Python code file that took 100 tokens in LLaMA 2 might need only 70–80 in LLaMA 3.

### 8.5 Mistral (Mistral AI)

| Version | Algorithm | Vocab | Library |
|---|---|---|---|
| Mistral v0.1 / v0.2 | SentencePiece BPE | 32,000 | sentencepiece |
| Mistral v0.3 | BPE | 32,768 | tiktoken |
| Mixtral 8×7B | SentencePiece BPE | 32,000 | sentencepiece |
| Mistral Large / Nemo | BPE | 131,072 | tekken (tiktoken-based) |

Mistral's "Tekken" tokenizer (v3+) uses 131,072 tokens trained on 24 languages with roughly equal representation. It significantly outperforms cl100k_base on non-English languages.

### 8.6 Falcon (TII, 2023)

| Property | Value |
|---|---|
| Algorithm | BPE (via HuggingFace tokenizers) |
| Vocabulary size | 65,024 |
| Library | HuggingFace `tokenizers` |

Falcon uses a custom BPE vocabulary trained on its multilingual corpus (RefinedWeb). It does not use regex pre-splitting — a controversial choice that means numbers and words can merge across boundaries.

### 8.7 Other Notable Models

| Model | Algorithm | Vocab Size | Notes |
|---|---|---|---|
| BERT / RoBERTa | WordPiece | 30,522 | `##` prefix for non-initial subwords |
| T5 / mT5 | SentencePiece Unigram | 32,000 / 250,000 | mT5 covers 101 languages |
| GPT-NeoX | BPE | 50,254 | EleutherAI's GPT-J/NeoX |
| Phi-3 | tiktoken | ~32,064 | Microsoft; same tokenizer as LLaMA 3 |
| Gemma (Google) | SentencePiece | 256,000 | Large vocab for multilingual |
| Qwen (Alibaba) | tiktoken | 151,936 | Includes Chinese characters as tokens |
| DeepSeek | BPE | 100,015 | tiktoken-based |
| Gemini | SentencePiece | Not disclosed | Likely ~256K |

---

## 9. Code Walkthrough — This Repository

### 9.1 Project Structure

```
tokenizer_project/
├── tokenizers/
│   ├── __init__.py          # Public API exports
│   ├── base.py              # Abstract Tokenizer + shared utils (get_stats, merge)
│   ├── basic.py             # BasicTokenizer  — pure byte-level BPE
│   ├── regex.py             # RegexTokenizer  — BPE with regex pre-splitting
│   └── sentencepiece.py     # SentencePieceBPETokenizer + SentencePieceWrapper
├── tests/
│   └── test_tokenizers.py   # 24 unit + integration tests (all pass)
├── train.py                 # CLI training script
└── README.md                # This file
```

### 9.2 The `base.py` Building Blocks

Every tokenizer inherits from `Tokenizer` and uses two pure functions:

**`get_stats(ids)`** — Count adjacent pairs:

```python
def get_stats(ids, counts=None):
    counts = counts or {}
    for pair in zip(ids, ids[1:]):          # sliding window of width 2
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# Example:
get_stats([1, 2, 3, 1, 2])
# → {(1,2): 2, (2,3): 1, (3,1): 1}
```

**`merge(ids, pair, new_id)`** — Replace a pair in one pass:

```python
def merge(ids, pair, new_id):
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            result.append(new_id)
            i += 2          # skip both elements of the pair
        else:
            result.append(ids[i])
            i += 1
    return result

# Example:
merge([1, 2, 3, 1, 2], (1, 2), 99)
# → [99, 3, 99]
```

These two functions are the complete heart of BPE. Everything else is bookkeeping.

### 9.3 `BasicTokenizer` — How Training Works

```python
tok = BasicTokenizer()
tok.train("aaabdaaabac", vocab_size=256 + 3)
```

Internally:

```
ids = [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]   ← UTF-8 bytes of "aaabdaaabac"
      a   a   a   b   d    a   a   a   b   a   c

Merge 1: (97, 97) → 256   "aa"
  ids = [256, 97, 98, 100, 256, 97, 98, 97, 99]

Merge 2: (256, 97) → 257  "aaa"
  ids = [257, 98, 100, 257, 98, 97, 99]

Merge 3: (257, 98) → 258  "aaab"
  ids = [258, 100, 258, 97, 99]
```

Encoding "aaabdaaabac" with the trained tokenizer returns `[258, 100, 258, 97, 99]` — exactly the 5 tokens from the Wikipedia BPE example. This is the canonical correctness check from Karpathy's minBPE.

### 9.4 `RegexTokenizer` — Pre-splitting in Action

```python
import re
text = "Hello world 123!!"
chunks = re.findall(GPT4_SPLIT_PATTERN, text)
# → ["Hello", " world", " 123", "!!"]
```

BPE is then applied **within each chunk independently**. The number `123` will never merge with the space before it. The `!!` punctuation will never merge with letters.

This is why GPT-4 can reliably split numbers digit-by-digit when needed for arithmetic, while GPT-2 sometimes creates awkward multi-character number tokens.

### 9.5 Special Token Handling

```python
tok = RegexTokenizer()
tok.train(corpus, vocab_size=1000)
tok.register_special_tokens({"<|endoftext|>": 1000, "<|pad|>": 1001})

# Disallow special tokens silently
ids = tok.encode("hello <|endoftext|> world", allowed_special="none")
# → encodes <|endoftext|> as regular bytes

# Allow all special tokens
ids = tok.encode("hello <|endoftext|> world", allowed_special="all")
# → [token_ids..., 1000, token_ids...]

# Raise if unexpected special token found
ids = tok.encode("hello <|endoftext|>", allowed_special="none_raise")
# → ValueError!
```

This three-mode system mirrors tiktoken's design and prevents accidental prompt injection from user-supplied text that contains special tokens.

### 9.6 Save / Load

```python
tok.save("models/my_tokenizer")
# Creates: models/my_tokenizer.model  (JSON — machine readable)
#          models/my_tokenizer.vocab  (text — human readable)

new_tok = RegexTokenizer()
new_tok.load("models/my_tokenizer.model")
```

The `.model` file stores the pattern string, all merge pairs, and special tokens in JSON. The `.vocab` file is for inspection — each line is `[  id] token_string`.

### 9.7 Running the Tests

```bash
python -m pytest tests/ -v
```

Expected output: **24 passed, 6 subtests passed**. Tests cover:
- Encode/decode roundtrips for all three tokenizer variants
- Unicode, emoji, and multilingual text
- Special token handling (allow / deny / raise)
- Cross-boundary merge prevention (regex tokenizer)
- Save/load persistence
- Known BPE output (Wikipedia example)
- Compression ratio validation

### 9.8 Training from CLI

```bash
# Train a RegexTokenizer on your corpus
python train.py \
    --input  data/corpus.txt \
    --output models/my_bpe \
    --vocab_size 8000 \
    --type regex \
    --pattern gpt4 \
    --verbose

# Train a SentencePiece-style tokenizer
python train.py \
    --input  data/corpus.txt \
    --output models/my_sp \
    --vocab_size 32000 \
    --type sentencepiece
```

---

## 10. Special Tokens Demystified

Special tokens are reserved IDs that have semantic meaning to the model architecture, not BPE-encoded text. They are added **after** BPE training, at the highest ID values.

### 10.1 Common Special Tokens by Model

| Token | Models | Purpose |
|---|---|---|
| `<\|endoftext\|>` | GPT-2/3/4 | Marks document boundary in training data |
| `<s>` / `</s>` | LLaMA 1/2, T5 | Beginning / End of sequence |
| `<\|begin_of_text\|>` | LLaMA 3 | BOS equivalent |
| `<\|end_of_text\|>` | LLaMA 3 | Document end |
| `<\|eot_id\|>` | LLaMA 3 | End of turn (chat) |
| `[INST]` / `[/INST]` | Mistral v1 | Instruction markers |
| `<\|im_start\|>` / `<\|im_end\|>` | Qwen, Phi-3 | ChatML format |
| `[PAD]` | BERT | Padding for batch alignment |
| `[CLS]` | BERT | Classification token |
| `[SEP]` | BERT | Separator between sequences |
| `[MASK]` | BERT | Masked language model target |
| `<\|fim_prefix\|>` etc. | GPT-4 | Fill-in-the-Middle (code completion) |
| `<tool_call>` etc. | Mistral, LLaMA 3 | Tool / function calling |

### 10.2 Why Special Tokens Are Dangerous

User input that contains special token strings can be misinterpreted:

```
User says:  "My name is <|endoftext|> and I hate..."
Model sees: [name_tokens..., 50256, ...]
```

The `<|endoftext|>` token (ID 50256) in the middle of a user prompt signals end-of-document to the model, potentially truncating its context or causing unexpected behaviour. This is why tiktoken / our RegexTokenizer requires explicit `allowed_special` declarations.

**Best practice:** Always sanitise or disallow special tokens in user-controlled input.

---

## 11. Vocabulary Size: How to Choose

### 11.1 Effects of Vocabulary Size

| Smaller vocab (e.g. 32K) | Larger vocab (e.g. 128K) |
|---|---|
| Smaller embedding table | Larger embedding table |
| Longer token sequences | Shorter sequences (more context efficient) |
| Slower multilingual coverage | Better multilingual and code coverage |
| Lower compute for output softmax | Higher compute for softmax |
| Better for resource-constrained models | Preferred for frontier models |

### 11.2 The Fertility Metric

**Fertility** = average number of tokens per word for a given tokenizer and corpus. Lower fertility = more efficient.

```
English text:   GPT-2 fertility ≈ 1.3   (1.3 tokens per word)
                LLaMA 3 fertility ≈ 1.1  (larger vocab → fewer tokens)

Chinese text:   GPT-2 fertility ≈ 4.5   (very bad — 3 bytes × 1.5 merges per char)
                LLaMA 3 fertility ≈ 1.8  (much better — dedicated CJK tokens)
```

### 11.3 Practical Guidance

| Use case | Recommended vocab size |
|---|---|
| English-only research model | 32,000 – 50,000 |
| General purpose English | 50,000 – 100,000 |
| Multilingual (European languages) | 64,000 – 128,000 |
| Multilingual (including CJK) | 128,000 – 256,000 |
| Code-heavy model | 64,000 – 128,000 |
| Production frontier model | 128,000+ |

---

## 12. Tokenizer Pathologies and Edge Cases

### 12.1 Numbers

One of the most well-known tokenizer quirks is inconsistent number tokenisation:

```python
# GPT-2 (50K vocab):
"1 + 1 = 2"   → ["1", " +", " 1", " =", " 2"]    ✅ digit-level
"100 + 100"   → ["100", " +", " 100"]              ✅ ok
"1000 + 1000" → ["1000", " +", " 1", "000"]        ❌ split mid-number!
"123456789"   → ["123", "456", "789"]               ❌ arbitrary splits
```

GPT-4's regex pre-pattern limits number chunks to 3 digits (`\p{N}{1,3}`), ensuring consistent tokenisation of numbers up to 999. Larger numbers are split into predictable 3-digit chunks.

LLMs are notoriously bad at arithmetic partly because of this — adding `1000 + 1000` when `1000` is sometimes 1 token and sometimes 2 is difficult for the model.

### 12.2 Whitespace and Indentation

Code models must handle indentation precisely:

```python
# "    def foo():" — 4 spaces of indentation
GPT-2:   [220, 220, 220, 220, 4299, ...]   # 4 separate space tokens — bad!
GPT-4:   [1881, 711, 34066, ...]            # "    def" as one token — efficient
```

This is why CodeLlama and similar code models often use larger vocabularies or dedicated tokenizer training on code corpora.

### 12.3 The "leading space" problem

Most BPE tokenizers encode the same word differently depending on whether it starts a sentence:

```python
tok.encode("Hello")          # → [9906]
tok.encode(" Hello")         # → [22691]   ← different token!
tok.encode("Say Hello")      # → [25, 22691]  ← "Hello" gets the space token
```

This can cause confusion when doing token-level operations. Always be consistent about whether you include leading spaces.

### 12.4 Case Sensitivity

```python
tok.encode("python")   # → [29, 7344]     (2 tokens)
tok.encode("Python")   # → [37, 4690]     (2 different tokens)
tok.encode("PYTHON")   # → [4464, 39, 1539, 1239]  (4 tokens!)
```

The same word in different cases produces completely different token sequences and lengths. This is why normalisation decisions during training matter — a model trained on mixed-case data must learn that all three refer to the same programming language.

### 12.5 Unicode Edge Cases

```python
# Visually identical but different Unicode code points:
"café"   (é = U+00E9)           → 3 tokens
"café"   (é = e + U+0301)       → 4 tokens   ← combining accent mark!

# Emoji with skin tone modifiers:
"👋"     → 3 tokens   (base emoji + skin tone + variation selector)
"👋🏽"    → 5 tokens
```

Always normalise text (e.g. NFC normalisation via `unicodedata.normalize("NFC", text)`) before tokenisation in production systems.

---

## 13. Multilingual Tokenization

### 13.1 The Allocation Problem

A fixed vocabulary of N tokens must cover all languages. How the vocabulary "budget" is split between languages matters enormously:

- A tokenizer trained only on English will assign most tokens to English subwords
- Chinese characters, being frequent in Chinese text, may each become a token — but only if the tokenizer has seen enough Chinese
- A language that gets few dedicated tokens has high fertility (many tokens per word) and therefore "uses up" more context window

### 13.2 Script-Level Comparison

```
English "tokenization" (1 word):
  GPT-2:   ["token", "ization"]                    = 2 tokens
  LLaMA3:  ["token", "ization"]                    = 2 tokens

Chinese "分词" (tokenization, 2 characters):
  GPT-2:   [168, 245, 167, 165]                    = 4 tokens (raw bytes!)
  LLaMA3:  [58200, 100004]                         = 2 tokens (dedicated CJK)
  Qwen:    [9101, 21188]                           = 2 tokens (large CJK vocab)

Arabic "ترميز" (tokenization, 5 chars):
  GPT-2:   10+ tokens (each Arabic byte encoded)
  LLaMA3:  3-4 tokens
```

### 13.3 Strategies for Multilingual Tokenizers

**Strategy 1: Language-balanced training corpus**
Train BPE on data with equal representation per language. This prevents English domination of the vocabulary.

**Strategy 2: Larger vocabulary**
More tokens → more room for each language. Gemma uses 256K, covering 100+ languages reasonably well.

**Strategy 3: Byte fallback**
Any character not in the vocabulary is encoded as its raw bytes (`<0xE4>`, etc.). This guarantees coverage but produces long sequences for unseen scripts.

**Strategy 4: SentencePiece Unigram with subword regularisation**
Training on multilingual data with regularisation produces more robust cross-lingual representations.

---

## 14. Training Your Own Tokenizer

### 14.1 When You Need a Custom Tokenizer

- Domain-specific corpus (medical, legal, code) where standard tokenizers are inefficient
- New language not well-covered by existing tokenizers
- You need a specific vocabulary size for hardware efficiency
- You want full control over special tokens and vocabulary composition

### 14.2 Using This Repository

```python
from tokenizers import RegexTokenizer, GPT4_SPLIT_PATTERN

# Load your corpus
with open("my_corpus.txt", "r") as f:
    corpus = f.read()

# Train
tok = RegexTokenizer(pattern=GPT4_SPLIT_PATTERN)
tok.train(corpus, vocab_size=16384, verbose=True)

# Evaluate fertility on a held-out set
test_text = "The quick brown fox jumps over the lazy dog."
ids = tok.encode(test_text)
words = test_text.split()
fertility = len(ids) / len(words)
print(f"Fertility: {fertility:.2f} tokens/word")

# Save
tok.save("models/custom_tokenizer")
```

### 14.3 Using the HuggingFace Tokenizers Library

For production-scale training (billions of tokens), use HuggingFace's Rust-based `tokenizers` library:

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["corpus.txt"],
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
)
tokenizer.save_model("models/")

# Convert to HuggingFace fast tokenizer
from transformers import PreTrainedTokenizerFast
hf_tok = PreTrainedTokenizerFast(tokenizer_file="models/tokenizer.json")
```

### 14.4 Using the Official SentencePiece CLI

```bash
# Install
pip install sentencepiece

# Train (for LLaMA-style model)
spm_train \
    --input=corpus.txt \
    --model_prefix=tokenizer \
    --vocab_size=32000 \
    --model_type=bpe \
    --character_coverage=0.9995 \
    --pad_id=3 \
    --bos_id=1 \
    --eos_id=2 \
    --unk_id=0 \
    --byte_fallback=true

# Outputs: tokenizer.model, tokenizer.vocab
```

### 14.5 Tokenizer Evaluation Checklist

After training, verify:

- [ ] **Roundtrip integrity:** `decode(encode(text)) == text` for 1000+ random strings
- [ ] **Fertility on held-out set:** Compare against target (e.g. ≤ 1.5 tokens/English word)
- [ ] **Special token handling:** Confirm BOS/EOS/PAD appear correctly
- [ ] **OOV handling:** Test with rare Unicode, emoji, mixed scripts
- [ ] **Compression ratio:** Encoded length should be shorter than raw bytes
- [ ] **Boundary preservation:** Numbers shouldn't merge with adjacent letters (regex tokenizer)
- [ ] **Consistency:** Same word should tokenise consistently regardless of position

---

## 15. Quick Reference: Model-to-Tokenizer Map

| Model family | Tokenizer | Vocab | Library | Pattern |
|---|---|---|---|---|
| GPT-2 | Byte-level BPE | 50,257 | tiktoken `gpt2` | GPT-2 regex |
| GPT-3 | Byte-level BPE | 50,257 | tiktoken `p50k_base` | GPT-2 regex |
| GPT-3.5 / text-davinci | Byte-level BPE | 50,281 | tiktoken `p50k_edit` | GPT-2 regex |
| GPT-4 | Byte-level BPE | 100,256 | tiktoken `cl100k_base` | GPT-4 regex |
| GPT-4o | Byte-level BPE | 200,019 | tiktoken `o200k_base` | GPT-4o regex |
| LLaMA 1 | SP BPE | 32,000 | sentencepiece | — |
| LLaMA 2 | SP BPE + byte fallback | 32,000 | sentencepiece | — |
| LLaMA 3 / 3.1 | Byte-level BPE | 128,000 | tiktoken | GPT-4 regex |
| Mistral v0.1/v0.2 | SP BPE | 32,000 | sentencepiece | — |
| Mistral v0.3+ / Nemo | Byte-level BPE | 131,072 | tekken | Custom |
| Mixtral 8×7B | SP BPE | 32,000 | sentencepiece | — |
| Falcon 7B/40B | BPE (HF) | 65,024 | HuggingFace tokenizers | None |
| BERT / RoBERTa | WordPiece | 30,522 | HuggingFace tokenizers | WordPiece |
| T5 | SP Unigram | 32,000 | sentencepiece | — |
| mT5 | SP Unigram | 250,000 | sentencepiece | — |
| Gemma | SP BPE | 256,000 | sentencepiece | — |
| Phi-3 Mini | tiktoken | 32,064 | tiktoken | LLaMA 3 regex |
| Qwen 1.5 / 2 | BPE (tiktoken-style) | 151,936 | tiktoken | Custom |
| DeepSeek | BPE | 100,015 | tiktoken | Custom |
| CodeLlama | SP BPE | 32,016 | sentencepiece | — |
| StarCoder | BPE (HF) | 49,152 | HuggingFace tokenizers | — |

---

## 16. Key Papers and Resources

### Foundational Papers

| Paper | Year | Contribution |
|---|---|---|
| Gage — "A New Algorithm for Data Compression" | 1994 | Original BPE algorithm |
| Sennrich, Haddow, Birch — "Neural Machine Translation of Rare Words with Subword Units" | 2016 | BPE for NLP/NMT |
| Kudo & Richardson — "SentencePiece: A simple and language independent subword tokenizer" | 2018 | SentencePiece system |
| Kudo — "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates" | 2018 | Unigram LM + regularisation |
| Radford et al. — "Language Models are Unsupervised Multitask Learners" (GPT-2) | 2019 | Byte-level BPE + regex pre-split |
| Brown et al. — "Language Models are Few-Shot Learners" (GPT-3) | 2020 | Scaled BPE vocabulary |
| Touvron et al. — "LLaMA: Open and Efficient Foundation Language Models" | 2023 | SentencePiece BPE at scale |
| Touvron et al. — "LLaMA 2: Open Foundation and Fine-Tuned Chat Models" | 2023 | SP BPE + byte fallback |

### Code and Libraries

| Resource | URL | Notes |
|---|---|---|
| tiktoken (OpenAI) | https://github.com/openai/tiktoken | Fast Rust BPE, GPT-2/4 tokenizers |
| sentencepiece | https://github.com/google/sentencepiece | C++ library, Python bindings |
| HuggingFace tokenizers | https://github.com/huggingface/tokenizers | Rust backend, many algorithms |
| minBPE (Karpathy) | https://github.com/karpathy/minbpe | Clean minimal BPE implementation |
| BPE from scratch (Raschka) | https://sebastianraschka.com/blog/2025/bpe-from-scratch.html | Step-by-step tutorial |
| Tiktokenizer (interactive) | https://tiktokenizer.vercel.app | Visual token explorer |

### Lectures and Tutorials

| Resource | Notes |
|---|---|
| Andrej Karpathy — "Let's build the GPT Tokenizer" | 2h YouTube lecture, highly recommended |
| Sebastian Raschka — "BPE from Scratch" | Blog post with full code walkthrough |
| HuggingFace NLP Course — Chapter 6 | Tokenizers deep-dive with exercises |
| "The Tokenizer Summit" at NeurIPS 2023 | Panel discussion on future of tokenisation |

---

## 17. Deep Concepts: Tokenization and Model Cognition

This section goes beyond "how tokenizers work" into the more subtle question: **how does the choice of tokenization shape what a model can and cannot think?**

### 17.1 Tokens are the Atoms of Thought

An LLM generates text one token at a time. Each forward pass produces a probability distribution over the entire vocabulary — the next token is selected from that distribution. This means **the token is the smallest unit of computation the model can operate on**.

Consider what this implies:

- The model cannot produce a word it has never seen as a single token unless it can spell it out sub-token by sub-token.
- Two concepts that share a token surface form (polysemy) force the model to disambiguate via context alone.
- A concept that requires many tokens to express costs proportionally more attention computation than a concept expressible in one.

This is not a metaphor. It is a mathematical constraint. The embedding matrix has shape `(vocab_size × d_model)`. Every token maps to exactly one row. The model has no mechanism to "look inside" a token — the raw bytes that compose it are invisible once encoded.

### 17.2 The Byte-Level Hypothesis

A recurring question in the field: *should we just skip tokenization entirely and work at the raw byte level?*

The argument for bytes:
- No vocabulary design decisions
- No OOV problem
- Perfectly consistent across languages
- Eliminates all tokenizer-related pathologies (numbers, punctuation, whitespace)

The argument against:
- Sequence length explodes. "Hello, world!" = 13 bytes vs 4 GPT-4 tokens. With a 4096-token context window, you could fit only ~315 words at byte level.
- The model must learn to reconstruct the character-level structure of language from scratch. This is possible but sample-inefficient — you need far more training data to learn the same linguistic patterns.
- Attention is O(N²) in sequence length. Byte-level sequences are 3–5× longer than BPE sequences. The compute cost increases by 9–25×.

**ByT5 (Google, 2022)** showed that byte-level models can match subword models on many tasks, especially morphologically rich languages, but at significantly higher compute cost per sample.

**MegaByte (Yu et al., 2023)** proposed a patch-level + byte-level hierarchical architecture to address the sequence length problem: a small model handles individual bytes within a "patch" (e.g. 4 bytes), while a larger model handles patches. This is one of the most promising tokenizer-free architectures to date.

### 17.3 Token Boundary Bias

Here is a subtle and underappreciated phenomenon: **the model learns the statistical distribution of what comes after token boundaries, not what comes after character boundaries.** This creates systematic biases.

**Example — reversal tasks:**

```
"Spell 'tiger' backwards"  →  expected: "r-e-g-i-t"
```

GPT-2 tokenizes "tiger" as a single token `[28333]`. The model has no direct access to the individual characters t-i-g-e-r — they are invisible sub-token structure. To reverse the spelling, it must have memorised from training data that the bytes of the "tiger" token happen to spell t-i-g-e-r. This is why LLMs famously struggle with character-level tasks:

```
"How many 'r's are in 'strawberry'?"  →  models often say 2 instead of 3
```

GPT-4 tokenizes "strawberry" as `["st", "raw", "berry"]`. The character `r` appears at the boundary between `raw` and `berry` — and the boundary itself is invisible to the model without explicit reasoning. With chain-of-thought prompting ("spell it out letter by letter"), models do far better because they are forced to generate each character token explicitly.

**The fix in modern models:** GPT-4o and o1-series models use a much larger vocabulary with character-level granularity for many common words. Gemini Ultra was reportedly trained with character-awareness objectives alongside standard language modelling.

### 17.4 Token Healing

**Token healing** is a technique introduced by Microsoft (used in Guidance, now in some production systems) to fix a subtle inference-time bug.

**The problem:** Suppose your prompt ends mid-word: `"The capital of France is Par"`. GPT-4 tokenizes this as `["The", " capital", " of", " France", " is", " Par"]`. The model now predicts what follows `Par` — but the token `" Paris"` (with leading space) is one token in the vocabulary, while `"is"` is the continuation of `Par`. The model must generate `"is"` then `" "` then other tokens, never naturally producing the efficient `" Paris"` token.

**The solution:** Before generation, remove the last partial token from the context and back up the generation to the last clean token boundary. Then re-generate from there, now allowing the model to produce `" Paris"` as a single efficient token.

```
Naïve:    prompt = "...Par"  →  model generates "is", " ", "a", " ", "beautiful"...
Healed:   prompt = "... is"  →  model generates " Paris", " is", " beautiful"...
```

Token healing is implemented in the `guidance` library and is particularly important for code completion, where prompts frequently end at arbitrary character positions.

### 17.5 Tokenization and Few-Shot Learning

The format of your few-shot examples interacts with tokenization in non-obvious ways.

**Consistent token boundaries across examples matter.** If your few-shot template is:

```
Input: Hello
Output: Bonjour

Input: Goodbye
Output:
```

The model learns a pattern where `Output:` is followed by a space and then the translation. But `" Bonjour"` (with leading space) and `"Bonjour"` (without) are different tokens with different embeddings. If your examples inconsistently use spaces, the model receives noisy in-context supervision.

**Tip:** Always verify your few-shot prompts using a tokenizer visualizer (e.g. tiktokenizer.vercel.app) to confirm that all analogous positions across examples land on consistent token boundaries.

---

## 18. WordPiece and Unigram LM — Full Derivations

These two algorithms are less commonly discussed than BPE but power BERT, T5, mT5, ALBERT, and many production systems.

### 18.1 WordPiece (Schuster & Nakamura, 2012; popularised by BERT)

WordPiece is BPE's cousin with one key difference: instead of choosing the **most frequent** pair to merge, it chooses the pair that **maximises the likelihood of the training data under a unigram language model**.

**The scoring criterion:**

Given the current vocabulary V and training corpus, the score of merging tokens A and B is:

```
score(A, B) = freq(AB) / (freq(A) × freq(B))
```

This is the **pointwise mutual information (PMI)** between A and B. It penalises merges where both A and B are themselves very frequent — because merging two already-common tokens gives less "new information" than merging two rarer tokens that almost always co-occur.

**Worked example:**

```
Suppose: freq("un") = 1000,  freq("##able") = 800,  freq("un##able") = 600

BPE score:       600          (just frequency — would rank highly)
WordPiece score: 600 / (1000 × 800) = 0.00075  (penalised by high individual freqs)

Compare: freq("hyper") = 10,  freq("##bolic") = 8,  freq("hyper##bolic") = 7

WordPiece score: 7 / (10 × 8) = 0.0875  (much higher! rarely seen separately)
```

WordPiece correctly prefers to merge `hyper` + `##bolic` (they almost always appear together) over `un` + `##able` (each is common independently).

**The `##` prefix convention:**

WordPiece uses `##` to mark subword tokens that are not word-initial. This is opposite to SentencePiece's `▁` which marks word-initial tokens. So:

```
BERT:         "tokenization" → ["token", "##ization"]
SentencePiece: "tokenization" → ["▁token", "ization"]
```

In BERT, the `[CLS]` token at position 0 is used for classification — the model is trained with an auxiliary task (next sentence prediction) that forces `[CLS]` to encode sentence-level semantics. This is purely a tokenizer-driven architectural choice.

### 18.2 Unigram Language Model (Kudo, 2018)

The Unigram LM algorithm is mathematically the most principled of all tokenization algorithms. It directly optimises a probabilistic objective.

**The objective:**

Assume the tokenization of a string `x` into tokens `(t₁, t₂, ..., tₙ)` has probability:

```
P(x) = P(t₁, t₂, ..., tₙ) = ∏ᵢ p(tᵢ)
```

This is the **unigram assumption**: each token is independent. The probability of the most likely tokenization of `x` is:

```
P*(x) = max over all segmentations T of x:  ∏ᵢ p(tᵢ)
```

This can be computed efficiently with the **Viterbi algorithm** (dynamic programming over the character lattice).

**Training procedure:**

```
1. Initialise: seed vocabulary = all substrings of length ≤ L occurring > threshold times
               Estimate p(tᵢ) = freq(tᵢ) / total_token_count  (EM initialisation)

2. E-step: For each sentence, find the most probable segmentation using Viterbi
            Compute expected counts E[tᵢ] = sum over sentences of (expected uses of tᵢ)

3. M-step: Update probabilities
            p(tᵢ) = E[tᵢ] / Σⱼ E[tⱼ]

4. Prune: Compute loss_i = reduction in log-likelihood if token i were removed
           Remove the bottom X% of tokens by loss_i
           (Typically X = 10–20% per iteration)

5. Repeat steps 2–4 until |vocab| = target size
```

**Why it's more principled than BPE:**

- BPE is greedy: each merge is locally optimal. There is no guarantee that the final vocabulary maximises any global objective.
- Unigram LM has a clear objective: maximise ∑ₓ log P*(x) over all training sentences.
- BPE tokenization is deterministic: one string → exactly one tokenization. Unigram LM can enumerate all valid tokenizations, enabling **subword regularisation**.

**Subword regularisation in practice:**

During model training (not tokenizer training), instead of always using the Viterbi (most probable) segmentation, we **sample** from the full distribution of valid segmentations:

```python
# Viterbi (deterministic) — used at inference time
"tokenization" → ["▁token", "ization"]   (always)

# Sampled (stochastic) — used during training
"tokenization" → ["▁token", "i", "z", "ation"]   (sometimes)
"tokenization" → ["▁t", "oken", "ization"]         (sometimes)
"tokenization" → ["▁token", "ization"]             (most often)
```

This acts as data augmentation: the model sees the same sentence tokenised many different ways, forcing it to learn representations that are robust to segmentation variation. Kudo (2018) showed this improves BLEU scores on low-resource translation by 1–2 points.

---

## 19. Information-Theoretic View of Tokenization

This section reframes tokenization using entropy and coding theory — the deepest mathematical lens available.

### 19.1 Tokenization as a Code

From an information-theoretic standpoint, a tokenizer is a **variable-length code** that maps a stream of bytes to a shorter stream of token IDs. The study of optimal variable-length codes is the subject of **source coding theory**.

**Shannon's Source Coding Theorem** (1948):

For a source with entropy H bits/symbol, no lossless code can achieve an average code length shorter than H bits/symbol. The optimal code length for symbol i is:

```
L(i) = -log₂ P(i)   bits
```

where P(i) is the probability of symbol i.

This is exactly what Huffman coding achieves. BPE is an approximation: by merging frequent pairs into single tokens, we are implicitly creating a code where frequent byte sequences have shorter representations (fewer tokens).

### 19.2 Entropy of Natural Language

**Character-level entropy of English** has been measured empirically at approximately **1.3 bits/character** (Shannon, 1951 — his famous "prediction and entropy of printed English" experiment). This means a perfectly optimal encoder could represent English text in 1.3 bits per character.

**Compare tokenizer efficiencies:**

```
Raw bytes (ASCII):       8.0 bits/character   (fixed-length code — very wasteful)
Huffman on characters:   ~4.1 bits/character  (optimal single-char code)
GPT-2 BPE (50K vocab):  ~2.0 bits/character  (good compression)
GPT-4 BPE (100K vocab): ~1.6 bits/character  (better)
Theoretical optimum:     ~1.3 bits/character
```

A larger vocabulary brings the tokenizer closer to the Shannon entropy limit, which is why larger vocabularies improve efficiency — they allow more specific codes for more frequent byte sequences.

### 19.3 Mutual Information Between Adjacent Tokens

One way to evaluate tokenizer quality: measure the **mutual information** between consecutive tokens in encoded text.

```
I(Tₙ; Tₙ₊₁) = Σᵢ Σⱼ P(tᵢ, tⱼ) × log[P(tᵢ, tⱼ) / (P(tᵢ) × P(tⱼ))]
```

A good tokenizer produces tokens that are **maximally informative about each other** — meaning the sequence has high mutual information between positions. Why? Because the language model's job is to predict the next token. Higher mutual information between consecutive tokens means there is more signal for the model to exploit, leading to lower perplexity and better learning.

BPE directly maximises this: by merging the most frequent (i.e. highest joint probability) pair, each merge step increases the mutual information between the resulting token and its neighbours relative to the baseline of treating each byte independently.

### 19.4 Perplexity Is Tokenizer-Relative

This is a critical and often missed point: **perplexity numbers are not comparable across models with different tokenizers.**

Perplexity is defined as:

```
PPL = exp(-1/N × Σᵢ log P(tᵢ | t₁, ..., tᵢ₋₁))
```

where N is the **number of tokens**, not the number of words. A model with a larger vocabulary will produce shorter tokenizations (fewer tokens) for the same text, which mechanically changes the perplexity value even if the underlying language model quality is identical.

**Concrete example:**

```
Text: "The quick brown fox"
GPT-2 (50K vocab): ["The", " quick", " brown", " fox"]  → 4 tokens → PPL computed over 4
LLaMA 3 (128K):    ["The", " quick", " brown", " fox"]  → still 4 here, but...

"Antidisestablishmentarianism"
GPT-2:   ["Anti", "dis", "establishment", "arian", "ism"]  = 5 tokens
LLaMA3:  ["Anti", "dis", "establish", "ment", "arian", "ism"] = 6 tokens
```

The model with fewer tokens per word has lower average perplexity all else equal, because it has fewer prediction steps (and thus fewer opportunities to accumulate prediction error). The correct comparison metric is **bits-per-character (BPC)** or **bits-per-byte (BPB)**, which normalises by the raw character count rather than the token count.

```
BPC = (sum of negative log-probs over tokens) / (number of characters)
```

Any paper comparing models across different tokenizers should use BPC/BPB, not raw perplexity.

### 19.5 Conditional Entropy and Token Predictability

The **conditional entropy** H(Tₙ₊₁ | Tₙ, Tₙ₋₁, ...) measures how unpredictable the next token is given all previous context. For a perfect language model, the conditional entropy equals the true entropy of the language — approximately 1.3 bits/character for English.

The gap between a model's effective conditional entropy and the theoretical optimum tells you how much room for improvement remains. Modern LLMs operating at perplexity ~3–5 on held-out text are achieving roughly 1.5–2.3 bits/token — close to the theoretical limit for token-level prediction, which is why further scaling returns diminishing perplexity improvements.

---

## 20. Tokenization as a Lossy Compression Problem

This framing is non-standard but illuminating: tokenization is **lossy** in a subtle way that most treatments miss.

### 20.1 What Information Is Destroyed

When text is tokenized, three kinds of information are discarded:

**1. Sub-token character structure**

The token `"tokenization"` (if it exists as a single token) carries no information about its constituent characters t-o-k-e-n-i-z-a-t-i-o-n. That structure is collapsed into a single integer ID. The model can only recover it by generalising from other tokens that share prefixes/suffixes (`"token"`, `"ization"`, etc.).

**2. Morphological boundaries**

Human readers know that "un-break-able" has three morphemes: a negation prefix, a root, and an adjective suffix. A tokenizer might split it as `["unbre", "akable"]`, destroying the morphological boundary. The model must learn this boundary from context alone — a harder inductive task.

**3. Typographic intent**

Consider:
```
"Hello" (quotation marks)
Hello   (no marks)
HELLO   (emphasis)
h e l l o  (deliberate spacing)
```

All of these communicate different things to a human reader. After tokenization and embedding, these distinctions may be compressed into nearby but not identical embedding vectors — or may map to the same token (for `Hello` vs `"Hello"` in some vocabularies).

### 20.2 The Reconstruction Test

A useful way to think about tokenizer quality: **given only the token IDs, how well can you reconstruct the original byte stream?**

For any lossless tokenizer, the answer should be: perfectly. This is the basic roundtrip guarantee. But "lossless" here means byte-level fidelity — not semantic fidelity.

Two very different texts can have identical token sequences if they happen to be tokenized to the same IDs. This cannot happen with correct byte-level BPE (decoding is injective by construction), but it can happen if you make mistakes in the encode/decode pipeline (e.g. normalising Unicode before encoding).

**UTF-8 normalisation forms and tokenization:**

```python
import unicodedata

s1 = "café"   # é as single code point U+00E9
s2 = "café"   # é as e + combining acute accent U+0301

# These look identical but have different byte representations
s1.encode("utf-8")  # b'caf\xc3\xa9'        — 5 bytes
s2.encode("utf-8")  # b'cafe\xcc\x81'        — 6 bytes

# After NFC normalisation, they become identical
unicodedata.normalize("NFC", s2) == s1   # True
```

If you normalise input before encoding but not during decoding, or vice versa, you introduce invisible asymmetry. Always apply `unicodedata.normalize("NFC", text)` at exactly one point in your pipeline, consistently.

### 20.3 The Compression-Generalisation Trade-off

There is a deep tension in tokenizer design between **compression** (fewer tokens per string) and **generalisation** (the model sees enough variation to learn robust patterns).

**Extreme compression** (very large vocabulary, many single-token words): The model rarely sees novel combinations. Most sentences are compressed into a small number of very specific tokens. The model may become overfit to common phrasings.

**Minimal compression** (character or byte level): The model sees enormous variation in raw symbol sequences. It must learn to compose characters into words, words into phrases, from scratch. This requires far more parameters and data.

**The optimal point** is language-and-task-dependent, which is why one-size-fits-all vocabularies of 32K–128K tokens dominate: they represent an empirically validated sweet spot for most natural language tasks.

---

## 21. Tokenizer-Model Co-design: The Hidden Coupling

The tokenizer and the model are not independent components. Decisions made during tokenizer design propagate into the model's internal representations in ways that are rarely made explicit.

### 21.1 Embedding Initialisation and Vocabulary Size

The embedding matrix is the first layer of every transformer LLM:

```
E ∈ ℝ^(V × d_model)
```

where V = vocabulary size, d_model = embedding dimension. Each row is a learnable vector — the "meaning" of that token.

**Key insight:** New tokens start with random embeddings. During training, their embeddings are updated by gradient descent only when they appear in the training batch. A token that appears very rarely will receive few gradient updates and therefore have a poorly trained embedding at the end of training.

This means vocabulary design directly affects embedding quality:

- Tokens that appear < 100 times in the training corpus will have nearly-random embeddings at the end of training, regardless of model size.
- Adding a large number of domain-specific tokens (e.g. 10,000 medical terms) to a general-purpose vocabulary will hurt those tokens unless you train on a proportionally large medical corpus.
- When fine-tuning a model with new special tokens (e.g. `[TOOL_CALL]`), their embeddings must be initialised carefully. A common technique: initialise from the mean of embeddings of semantically related tokens (e.g. average of `"tool"`, `"function"`, `"call"`).

### 21.2 The Unembedding Layer (LM Head)

The output of the transformer is projected back to vocabulary space through the **LM head** (also called the unembedding matrix):

```
U ∈ ℝ^(d_model × V)
```

In most modern LLMs (GPT-2, LLaMA, Mistral), the LM head **shares weights** with the input embedding matrix: U = Eᵀ. This is called **weight tying** and was introduced in "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017).

Weight tying has several consequences:

- The model must simultaneously use the same matrix to *look up* a token's representation (forward) and to *score* all tokens as candidates for the next position (backward). This creates a consistency constraint that regularises learning.
- The inner product `Eᵀ × h` (hidden state dotted with each token embedding) gives the logit for each token. Tokens with similar embeddings will have similar logits in similar contexts — a desirable property.
- Adding new tokens to the vocabulary requires extending both E and U simultaneously and keeping them in sync.

### 21.3 Token Frequency and Gradient Flow

During training, the gradient with respect to a token's embedding is proportional to how often that token appears in the loss computation. Concretely, if token i appears Nᵢ times in a training epoch, its embedding receives Nᵢ gradient updates. For a Zipf-distributed vocabulary:

```
P(rank r) ∝ 1/r

Token rank 1 (most common, e.g. " the"):   ~5% of all tokens
Token rank 1000:                             ~0.005% of all tokens
Token rank 50000 (rare subword):             ~0.0001% of all tokens
```

The rarest tokens in a 50K vocabulary receive orders of magnitude fewer training signal updates than common tokens. This is why vocabulary pruning matters: tokens below some frequency threshold contribute noise, not signal, to the embedding table.

**Practical implication:** When training a model from scratch, it is better to use a slightly smaller vocabulary where every token appears frequently than a large vocabulary with a long tail of rare tokens. When fine-tuning, avoid adding tokens that appear fewer than a few thousand times in your fine-tuning data.

### 21.4 Tokenization and Positional Encoding

Positional encodings assign a unique signal to each position in the token sequence. Every transformer uses some form of positional encoding — absolute sinusoidal (original Transformer), learned absolute (GPT-2), relative (T5), or rotary (RoPE, used in LLaMA/Mistral).

**The critical coupling:** positional encodings are defined in terms of **token positions**, not character positions. Position 5 means "the 5th token", regardless of how many characters that token represents.

This matters for tasks where the character-level or word-level position is relevant (e.g. counting, copying, structured formatting). The model has no positional information about where in a token it is — only where the token is in the sequence. This is another manifestation of the sub-token invisibility problem from Section 17.3.

**RoPE (Rotary Position Embedding)** and its extension to long contexts (YaRN, LongRoPE) are defined in token-position space. When a model is adapted from a 4K token context to a 128K token context, the positional encodings must be scaled — but "128K tokens" corresponds to very different numbers of words depending on the tokenizer used.

### 21.5 Vocabulary Transfer: Moving Tokenizers Between Models

A common practical problem: you want to use a pretrained model but with a different tokenizer (e.g. extending LLaMA's 32K vocab to 128K for better multilingual coverage).

The standard technique is **vocabulary transfer** (also called tokenizer extension):

```
Step 1: Train new tokenizer with target vocab size on desired corpus

Step 2: For each token in the new vocab:
          If it exists in old vocab: copy the embedding row
          If it's new: initialise as mean of old-vocab token embeddings
                       that overlap with it byte-by-byte

Step 3: Continue pretraining on multilingual data to adapt the new embeddings

Step 4: Fine-tune for target tasks
```

This technique was used to create **Chinese-LLaMA** (extended LLaMA 1 with 20K Chinese tokens), **Llama-2-Ko** (Korean), and similar derivatives. The key finding: models with vocabulary extension adapt much faster in domain-specific fine-tuning because common domain terms are single tokens rather than 3–5 tokens.

---

## 22. Prompt Engineering at the Token Level

Understanding tokenization is a superpower for prompt engineering. Every prompt-writing decision is secretly a tokenization decision.

### 22.1 Token-Aware Prompt Formatting

**Rule 1: Count tokens, not characters.**

A 100-character limit is meaningless. A 100-token limit is a hard constraint. Different content types have wildly different token densities:

```
Type                          Approximate tokens per 1000 characters
English prose                 ~230 tokens
Python code                   ~280 tokens
Dense mathematical notation   ~400 tokens
JSON (with keys)              ~350 tokens
Minified JSON                 ~220 tokens
Base64-encoded data           ~350 tokens
Chinese text                  ~500 tokens (GPT-4, due to multi-byte chars)
```

**Rule 2: Prefer token-aligned delimiters.**

Some delimiters are single tokens; others are multiple tokens:

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

enc.encode("###")         # → [14468]         1 token  ✅ efficient
enc.encode("---")         # → [11192]         1 token  ✅ efficient
enc.encode("<separator>") # → [27, 25004, 29] 3 tokens ⚠️ less efficient
enc.encode("===")         # → [28, 1490]      2 tokens
```

When you control the prompt format, choose delimiters that are single tokens. This is not just about efficiency — it also makes the model's task easier, because it sees a clean boundary signal in one prediction step rather than three.

**Rule 3: Canonical spacing matters.**

```python
enc.encode("Answer:")      # → [16533, 25]       2 tokens
enc.encode(" Answer:")     # → [22559, 25]        2 tokens (different!)
enc.encode("\nAnswer:")    # → [198, 16533, 25]   3 tokens
enc.encode("\n\nAnswer:")  # → [271, 16533, 25]   3 tokens
```

The double newline `\n\n` is a single token in cl100k_base (`[271]`). This means `\n\nAnswer:` is 3 tokens, same as `\nAnswer:`. But `\n\n\nAnswer:` is 4 tokens. Models are sensitive to these invisible distinctions because they affect how the model has "seen" similar patterns during training.

### 22.2 The Priming Effect

**Token priming** is the observation that the last token in a prompt strongly conditions the first token of the response. This is a direct consequence of autoregressive generation: the first response token is sampled from `P(t | prompt_tokens)`, and the last prompt token is the most recent context.

**Practical consequence:** If you want the model to respond in a certain style, end your prompt with a token that primes that style:

```
"Please answer concisely:\n\n"   → primes for a short answer
"Step 1:"                         → primes for step-by-step reasoning
"```python\n"                      → primes for Python code
"The answer is:"                  → primes for a direct answer
```

This is why chain-of-thought prompting works so reliably — "Let's think step by step" ends with a token sequence that the model has seen thousands of times before multi-step reasoning in training data.

### 22.3 Context Budget Allocation

For long-context applications, understanding token costs enables better budget allocation:

```
GPT-4 context: 128,000 tokens

System prompt:            ~500 tokens  (0.4%)
User conversation:        ~2,000 tokens  (1.6%)
Retrieved documents:      ~120,000 tokens  (93.8%)
Response:                 ~5,500 tokens  (4.3%)
                          ────────────────────
Total:                    128,000 tokens
```

Knowing that a retrieved document chunk costs 300 tokens lets you determine you can retrieve exactly 400 chunks before hitting the context limit. Tools like `tiktoken` allow you to compute token counts before making API calls.

**Chunking strategy for RAG:**

The optimal chunk size for retrieval-augmented generation is typically 200–500 tokens per chunk — not 200–500 characters, not 200–500 words. Token-aligned chunking ensures that:

1. Chunk sizes are predictable relative to the context window.
2. Chunks do not break in the middle of a token (which would corrupt the text).
3. You can accurately predict how many chunks fit in a given context.

### 22.4 Logprobs and Token Probabilities

When you request `logprobs=True` from an API, you receive the log-probability of each generated token. This gives you a window into the model's confidence, but it must be interpreted carefully because it is token-level, not character- or word-level:

```
Generated:  " Paris"  →  logprob = -0.12  (very confident)
Generated:  " London" →  logprob = -2.8   (much less confident)
```

But `" Paris"` is a single token. The probability `exp(-0.12) ≈ 0.89` means the model was 89% confident that the entire string " Paris" (space + P + a + r + i + s) was the right next unit. Under a character-level model, the equivalent calculation would involve the product of 6 character probabilities — a fundamentally different quantity.

When computing token-level calibration or uncertainty estimates, always normalise by token length (in characters or bytes) to make comparisons meaningful.

---

## 23. Tokenization in Multimodal Models

Modern LLMs are increasingly multimodal — they process images, audio, video, and structured data alongside text. This requires extending the tokenization concept beyond strings.

### 23.1 Image Tokenization

**Patch-based approach (ViT-style):**

Vision Transformers (ViT) divide an image into a grid of fixed-size patches (e.g. 16×16 pixels each) and linearly project each patch to a d-model dimensional vector. For a 224×224 image with 16×16 patches:

```
Number of patches = (224/16)² = 196 image tokens
```

These 196 "image tokens" are concatenated with text tokens and processed by the same transformer. This is the approach used by LLaVA, Flamingo, and most open-source vision-language models.

**Discrete image tokens (VQ-VAE approach):**

An alternative: use a Vector-Quantised Variational Autoencoder (VQ-VAE) to map image patches to discrete integer codes from a learned codebook. Now image tokens are integers, just like text tokens — the same embedding table can represent both.

```
Image patch → CNN encoder → quantise to nearest codebook entry → integer ID
```

DALL-E (the original), CogView, and Parti use this approach. The entire image-text generation problem becomes a single sequence prediction problem.

**Tradeoffs:**

| Method | Tokens per image | Editability | Training data needed |
|---|---|---|---|
| ViT patches (continuous) | 196–1024 | Low | Moderate |
| VQ-VAE (discrete) | 256–1024 | High | Large |
| Pixel-level (raw) | 50K–250K | Highest | Enormous |

### 23.2 Audio Tokenization

**Spectrogram approach:**

Raw audio (16kHz, 16-bit) generates 16,000 samples per second. This is far too many to process directly. Most audio LLMs convert audio to a mel spectrogram and then either:

1. Use fixed-size time-frequency patches (analogous to ViT patches for images)
2. Use a neural audio codec (EnCodec, DAC, SoundStream) to produce discrete tokens

**EnCodec / RVQ approach:**

Residual Vector Quantisation (RVQ) uses a cascade of codebooks:

```
Audio frame → codec encoder → RVQ quantisation at N levels
             → N discrete tokens per frame, each from a codebook of size 1024
```

Meta's EnCodec at 24kHz with 8 codebook levels produces 8 tokens per audio frame (25 frames/second) = 200 tokens/second. A 10-second audio clip = 2000 tokens. This is manageable in a transformer context window.

**MusicGen, AudioCraft, and SpeechGPT** all use RVQ-based discrete audio tokenization. The key challenge is that N tokens are generated simultaneously per timestep (one per codebook level), which requires special generation strategies (delay patterns, parallel generation heads).

### 23.3 Video Tokenization

Video = images over time. Naive tokenization of 1080p video at 24fps:

```
Patches per frame: (1080/16)² = 4556 patches
Frames per second: 24
Tokens per second: 4556 × 24 = 109,344 tokens/second
```

This is completely intractable for a transformer with a 128K context window. Three techniques address this:

**1. Aggressive downsampling:** Reduce to 224×224 resolution, 2fps, with 32×32 patches → 14 tokens/frame × 2fps = 28 tokens/second. A 60-second clip = 1680 tokens.

**2. Temporal token merging:** Adjacent frames are highly similar. Techniques like TokenMerge (Bolya et al.) identify and merge near-duplicate tokens across time, reducing video token count by 40–60% with minimal quality loss.

**3. Hierarchical encoding:** A fast low-resolution model watches the full video and identifies key frames. A slow high-resolution model processes only those key frames in detail. This is the approach used in Gemini Ultra for long video understanding.

### 23.4 Unified Tokenization Spaces

The most elegant multimodal architectures use a **single token space** for all modalities:

```
Text:  → BPE tokens (IDs 0–131071)
Image: → VQ-VAE tokens (IDs 131072–163839)  [32768 image codebook entries]
Audio: → EnCodec tokens (IDs 163840–165887)  [2048 audio codebook entries]
```

The model learns cross-modal associations because text tokens and image tokens co-occur in training data (e.g. a caption and its image). This approach was used by DALL-E, GPT-4V (partially), and Chameleon (Meta, 2024).

**Chameleon (2024)** is the most complete publicly described example: a single transformer processes interleaved text and image tokens from a unified vocabulary of 65,536 entries. Text tokens are BPE from a 32K vocabulary; image tokens are VQ-VAE codes from a separate 8K codebook. The model can generate both text and images in any interleaved order.

---

## 24. Adversarial Tokenization and Security

Tokenization introduces attack surfaces that are largely invisible to end users and often overlooked by ML practitioners.

### 24.1 Tokenization-Based Prompt Injection

**The homoglyph attack:**

Unicode contains many characters that look identical to common ASCII characters but have different code points:

```
"a" (U+0061 Latin Small Letter A)         → tokenizes normally
"а" (U+0430 Cyrillic Small Letter A)       → different token ID!
"ɑ" (U+0251 Latin Small Letter Alpha)      → yet another token ID!
```

An attacker can craft a string that looks like `"Ignore previous instructions"` to a human reader but tokenizes completely differently to the model:

```
"Ignоre previоus instructiоns"   ← three o's are Cyrillic U+043E, not Latin U+006F
```

This text will pass through most text-based content filters (which check for ASCII strings) but may behave differently in the model because it tokenizes as an unusual sequence. Whether this produces the intended jailbreak effect depends on the model's training data, but the asymmetry between human-readable and machine-interpreted text is a genuine attack surface.

**Detection:** Always normalise input to NFC Unicode and optionally to ASCII-safe characters before running through a model if security is a concern.

### 24.2 The "Glitch Token" Phenomenon

In 2023, researchers discovered that GPT-2 (and to some extent GPT-3) had vocabulary entries that caused the model to behave erratically when they appeared as input — producing nonsensical outputs, refusing to repeat the token, or generating hallucinations.

**Root cause:** The BPE tokenizer for GPT-2 was trained on a corpus that included Reddit usernames and other internet artefacts. Some tokens in the vocabulary (e.g. `" SolidGoldMagikarp"`, a Reddit username) appeared in the *tokenizer training corpus* but were effectively never seen by the *language model* during its training (because the LM training data was filtered differently).

This means:
- The token exists in the vocabulary (it was learned during tokenizer training)
- The token has a randomly-initialised or minimally-trained embedding in the LM (it almost never appeared in LM training)
- When this token is presented to the LM, it has no sensible representation → the model extrapolates wildly

This is the "untrained embedding" problem from Section 21.1 manifesting at inference time. Modern tokenizers avoid this by training the tokenizer on a subset of the LM training data and pruning tokens below a frequency threshold.

### 24.3 Context Overflow via Token Amplification

**The token inflation attack:**

Some inputs are short in characters but long in tokens:

```python
# This string is 12 characters
text = "🏳️‍🌈🏳️‍⚧️"  

# But it tokenizes to many more tokens than expected
ids = enc.encode(text)
len(ids)  # Could be 20+ tokens for complex emoji with ZWJ sequences
```

An attacker who controls a small portion of the prompt (e.g. a username field or a product review) can craft strings that consume disproportionate context window space, potentially crowding out other important context or causing the system to truncate important instructions.

**Mitigation:** Always enforce token-level length limits, not character-level limits, for user-controlled inputs. Compute `len(enc.encode(user_input))` before inserting into the prompt.

### 24.4 Tokenization Inconsistency Between Train and Inference

A subtle but serious production bug: **if the tokenizer used during training differs from the tokenizer used during inference, the model sees a token distribution it was never trained on.**

This can happen in several ways:

1. **Version mismatch:** tiktoken releases occasionally change encoding details. A model trained with tiktoken 0.3.x may see different token IDs with tiktoken 0.6.x for edge-case inputs.

2. **Special token handling:** If the training pipeline handled `<|endoftext|>` as a literal string (not a special token) in some samples, the model learned to associate the text "endoftext" appearing between `<|` and `|>` with document boundaries. This is different from seeing the single special token ID.

3. **Stripping vs. not stripping:** Some training pipelines strip whitespace from the beginning/end of training documents. This changes whether the first word of a document has a leading space token (`" The"`) or not (`"The"`). If the convention differs at inference time, every prompt will start with a slightly wrong token.

**Best practice:** Pin your tokenizer version in `requirements.txt`, version-control your tokenizer configuration, and test tokenizer output on a fixed set of reference strings as part of your CI pipeline.

---

## 25. Future Directions: Tokenizer-Free and Dynamic Tokenization

### 25.1 The Case Against Fixed Tokenization

Every tokenizer discussed in this guide shares one property: the vocabulary is **fixed at training time** and never changes at inference time. This creates several fundamental limitations:

- A new technical term coined after tokenizer training cannot be represented as a single token.
- Rare languages are permanently disadvantaged by vocabulary allocation decisions made on the training corpus.
- The model cannot adapt its "resolution" based on the complexity of the input — every string gets the same fixed-granularity encoding.

### 25.2 MEGABYTE: Patch-Level Hierarchical Generation

**MegaByte** (Yu et al., 2023) proposes replacing the tokenizer with a hierarchical model:

```
Input bytes: [h, e, l, l, o, ,, ' ', w, o, r, l, d]
                ↓
Patch (4 bytes): [hell] [o,  ] [wor] [ld]
                ↓
Global model: processes patches, produces patch-level context
                ↓
Local model:  for each patch, generates bytes conditioned on global context
```

The "tokenizer" here is just a fixed-width splitter (every 4 bytes = one patch). No vocabulary training required. No BPE. No design decisions about merge order. The model learns to handle any byte sequence.

**Results:** MegaByte matches byte-level baseline models with 40% fewer FLOPs on language modelling. But it still lags behind BPE-based models in downstream task performance, suggesting that learned subword structure is genuinely useful, not just a computational convenience.

### 25.3 FLERT and Contextual Tokenization

**Contextual tokenization** is the idea that the same string should tokenize differently depending on context. For example, in medical text, "aspirin" should be a single token; in a phonics exercise, it should be `["as", "pir", "in"]`.

Current tokenizers are context-free: the encoding of "aspirin" is always the same regardless of what surrounds it. Contextual tokenizers would require:

1. A lightweight "tokenizer model" that reads context and decides on segmentation.
2. A mechanism for the LLM to know which tokenization was used for each position.

This is an open research problem. Early work includes **dynamic tokenization** (Mathis et al., 2023) and **character-aware tokenization** (used implicitly in some retrieval-augmented systems).

### 25.4 Entropy Coding as Tokenization

**Arithmetic coding** (the basis of gzip, LZMA, and modern neural data compression) can be used as a tokenizer:

The idea: use the LLM itself to compute the probability of each character, then use those probabilities to arithmetic-code the text into a bitstream. The "tokens" are variable-length bit patterns whose length is exactly -log₂ P(character | context). Decoding requires running the LLM forward pass for each character.

This is **tokenizer-free in the sense that the LM and tokenizer are fused**: the tokenizer is the LM. Neural data compression systems like **DeepMind's Chinchilla-based compressor** have achieved compression ratios surpassing gzip on natural language using this approach.

The practical limitation: you cannot train an LM this way in a single pass — it is circular. But for inference and evaluation, arithmetic coding provides a principled, tokenizer-free way to measure the true information content of a model's language predictions.

### 25.5 Learned Dynamic Vocabularies

Several recent papers propose **updating the vocabulary during training** rather than fixing it at the start:

**Vocabulary Expansion (FOCUS, 2023):** Start with a small vocabulary. As training proceeds, identify the most common token n-grams and add them to the vocabulary as single tokens. Periodically retokenize the training data with the updated vocabulary.

**BPE-Dropout (Provilkov et al., 2020):** During training, randomly omit some merges with probability p, producing a more fragmented tokenization. This is a form of subword regularisation for BPE (analogous to SentencePiece's built-in regularisation for Unigram LM) and consistently improves performance on low-resource languages by 0.5–1.5 BLEU.

```python
# BPE-Dropout: during training, randomly skip merges
import random

def encode_with_dropout(text, merges, dropout_p=0.1):
    ids = list(text.encode("utf-8"))
    for pair, new_id in sorted(merges.items(), key=lambda x: x[1]):
        if random.random() < dropout_p:
            continue   # randomly skip this merge
        ids = merge(ids, pair, new_id)
    return ids
```

**SpaceByte (Slagle, 2024):** A hybrid architecture that processes most text at the byte level but uses a larger patch for spaces (word boundaries), effectively recovering word-level structure without explicit tokenization. This performs comparably to BPE on English benchmarks while being strictly superior on morphologically complex languages.

### 25.6 What Will Replace BPE?

Based on current research trajectories, the most likely candidates are:

**Near-term (1–3 years):**
- Larger BPE vocabularies (200K–500K) with better multilingual coverage
- Byte-fallback becoming universal (already standard in newer models)
- Token healing becoming standard in production inference

**Medium-term (3–7 years):**
- Hierarchical byte/patch models supplementing or replacing BPE for new architectures
- Dynamic vocabulary updates during continued pretraining
- Tokenizer-model co-training where the tokenizer parameters are jointly optimised with the LM

**Long-term research horizon:**
- Fully tokenizer-free architectures operating at byte or bit level
- Arithmetic coding / neural compression-based sequence models
- Biological-inspired "chunking" mechanisms where segment boundaries are predicted dynamically

The fundamental tension will persist: **every bit of structure you bake into the tokenizer is a bit the model doesn't have to learn from data, but also a bit of flexibility you permanently sacrifice.** The ideal tokenizer is one that encodes exactly the structure present in the data — no more, no less — and this is ultimately a function of the task, the data, and the compute budget available.

---

### Lectures and Tutorials

| Resource | Notes |
|---|---|
| Andrej Karpathy — "Let's build the GPT Tokenizer" | 2h YouTube lecture, highly recommended |
| Sebastian Raschka — "BPE from Scratch" | Blog post with full code walkthrough |
| HuggingFace NLP Course — Chapter 6 | Tokenizers deep-dive with exercises |
| "The Tokenizer Summit" at NeurIPS 2023 | Panel discussion on future of tokenisation |

---

*This README is designed as a living reference. The tokenization landscape evolves quickly — new models may introduce new conventions not listed here.*
