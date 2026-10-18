#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = []
# ///
"""
Lesson 6.2: Implement Toy BPE (Byte-Pair Encoding).

You build the tokenizer behind GPT-2 in about sixty lines. Start with the 256
byte values as tokens, count every adjacent pair, merge the most frequent pair
into a new token, and repeat. The ordered list of merges is the whole model:
you encode by replaying the merges in the order they were learned, and decode
by expanding each token back to bytes. Text with characters the trainer never
saw still round-trips, because every byte is already a token.

Sources:

- Philip Gage, "A New Algorithm for Data Compression", C Users Journal 12(2),
  1994. BPE as compression: replace the most frequent pair of bytes with an
  unused byte.
- Sennrich, Haddow, Birch, "Neural Machine Translation of Rare Words with
  Subword Units", ACL 2016, arXiv:1508.07909. BPE for subword vocabularies.
- Radford et al., "Language Models are Unsupervised Multitask Learners"
  (GPT-2), 2019. Byte-level BPE with a base vocabulary of 256.
- karpathy/minbpe, https://github.com/karpathy/minbpe, a reference
  implementation.

Prerequisites (must complete first)
----------------------------------
- Lesson 6.1: Naive Tokenization Breaks Reality (regex tokenizers and where they break)

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. Train byte-pair merges by counting adjacent pairs and merging the most frequent.
2. Encode text by applying merges in rank order, and decode ids back to bytes.
3. Measure compression in bytes per token on held-out text, and show that a
   wrong merge order or a decoder that ignores a merge breaks the tokenizer.

The Big Idea
------------
BPE learns its vocabulary from data: frequent byte sequences become single
tokens, and everything else stays a run of bytes.

Why It Matters
--------------
A language model sees token ids, not text. The tokenizer fixes how long a
prompt is, what it costs, and which strings the model can represent at all.
Byte-level BPE never meets an unknown character, so there is no ``<unk>``
token, and every string has an encoding.

Stdlib Pain → Library Power-Up
------------------------------
Pain (this lesson):
- Each merge rescans the whole token list to count pairs, then rewrites it.
- Lines of code: ~60 for train, encode, and decode.
- Runtime: well under a second for the embedded 3.5 KB training text; megabytes
  of text and tens of thousands of merges take hours this way.
- Common bugs: applying merges in the wrong order, breaking ties
  nondeterministically, and forgetting that a merge can create new pairs.

Power-Up (Lesson 6.3 — tiktoken):
- ``tiktoken.get_encoding("gpt2")`` loads a trained merge table and applies it
  in compiled code, with a regex that stops merges crossing word boundaries.
- Trade-off: you give up the transparency of this file and train nothing
  yourself; you gain speed and a vocabulary that matches a real model.

Complexity Analysis
-------------------
Pain (manual), for n bytes of text and m merges:
- Training time: O(n·m). Every merge recounts all pairs in O(n) and rewrites
  the sequence in O(n).
- Encoding time: O(n·m) in the worst case, for the same reason.
- Decoding time: O(m + n·L), where L is the longest token in bytes.
- Space: O(n + m).

Execution
---------
Project-style:
    uv run python src/062_toy_bpe.py

Script-style (self-contained deps):
    uv run --script src/062_toy_bpe.py

Options:
    python src/062_toy_bpe.py --doctest-only
    python src/062_toy_bpe.py --no-doctest
    python src/062_toy_bpe.py --index
    python src/062_toy_bpe.py -v

Quality Gates
-------------
    uv run pytest src/062_toy_bpe.py
    uv run ruff check src/062_toy_bpe.py
    uv run ruff format --check src/062_toy_bpe.py
    uv run mypy --strict src/062_toy_bpe.py

Module Doctests
---------------
>>> merges = train("aaabdaaabac", 3)
>>> merges
[(97, 97), (97, 98), (256, 257)]
>>> ids = encode("aaabdaaabac", merges)
>>> ids
[258, 100, 258, 97, 99]
>>> decode(ids, merges) == b"aaabdaaabac"
True

Characters the trainer never saw fall back to raw bytes and still round-trip:

>>> odd = "café 世界"
>>> odd_ids = encode(odd, merges)
>>> len(odd_ids) == len(odd.encode())
True
>>> decode(odd_ids, merges).decode() == odd
True
"""

from __future__ import annotations

import argparse
import doctest
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON_ID: Final[str] = "tier_6_lesson_2_toy_bpe"

# The library form arrives with tiktoken.
POWER_UP_AVAILABLE: Final[bool] = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason

PREREQUISITES: Final[Mapping[str, str]] = {
    "tier_6_lesson_1_naive_tokenization": "Regex tokenizers and where they break",
}

DEPENDENCIES: Final[Mapping[str, str]] = {}

SUCCESSORS: Final[Mapping[str, str]] = {
    "tier_6_lesson_3_tiktoken": "A production byte-level BPE to compare against",
}

DEPENDENTS: Final[Mapping[str, str]] = {
    "tier_6_lesson_3_tiktoken": "Loads trained merges and applies them in compiled code",
}

RELATED: Final[Mapping[str, str]] = {}

POWER_UP_IN_LESSON: Final[str | None] = "tier_6_lesson_3_tiktoken"
PAIN_FROM_LESSON: Final[str | None] = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORPUS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Abraham Lincoln, Second Inaugural Address (1865). Text from the Avalon Project,
# Yale Law School, https://avalon.law.yale.edu/19th_century/lincoln2.asp
# Public domain: a speech of a U.S. president from 1865. Paragraphs are separated by
# blank lines and wrapped at spaces; ``split_corpus`` joins each onto one line.
CORPUS: Final[str] = """\
At this second appearing to take the oath of the Presidential office there is less occasion for
an extended address than there was at the first. Then a statement somewhat in detail of a course
to be pursued seemed fitting and proper. Now, at the expiration of four years, during which
public declarations have been constantly called forth on every point and phase of the great
contest which still absorbs the attention and engrosses the energies of the nation, little that
is new could be presented. The progress of our arms, upon which all else chiefly depends, is as
well known to the public as to myself, and it is, I trust, reasonably satisfactory and
encouraging to all. With high hope for the future, no prediction in regard to it is ventured.

On the occasion corresponding to this four years ago all thoughts were anxiously directed to an
impending civil war. All dreaded it, all sought to avert it. While the inaugural address was
being delivered from this place, devoted altogether to saving the Union without war, insurgent
agents were in the city seeking to destroy it without war--seeking to dissolve the Union and
divide effects by negotiation. Both parties deprecated war, but one of them would make war
rather than let the nation survive, and the other would accept war rather than let it perish,
and the war came.

One-eighth of the whole population were colored slaves, not distributed generally over the
Union, but localized in the southern part of it. These slaves constituted a peculiar and
powerful interest. All knew that this interest was somehow the cause of the war. To strengthen,
perpetuate, and extend this interest was the object for which the insurgents would rend the
Union even by war, while the Government claimed no right to do more than to restrict the
territorial enlargement of it. Neither party expected for the war the magnitude or the duration
which it has already attained. Neither anticipated that the cause of the conflict might cease
with or even before the conflict itself should cease. Each looked for an easier triumph, and a
result less fundamental and astounding. Both read the same Bible and pray to the same God, and
each invokes His aid against the other. It may seem strange that any men should dare to ask a
just God's assistance in wringing their bread from the sweat of other men's faces, but let us
judge not, that we be not judged. The prayers of both could not be answered. That of neither has
been answered fully. The Almighty has His own purposes. "Woe unto the world because of offenses;
for it must needs be that offenses come, but woe to that man by whom the offense cometh." If we
shall suppose that American slavery is one of those offenses which, in the providence of God,
must needs come, but which, having continued through His appointed time, He now wills to remove,
and that He gives to both North and South this terrible war as the woe due to those by whom the
offense came, shall we discern therein any departure from those divine attributes which the
believers in a living God always ascribe to Him? Fondly do we hope, fervently do we pray, that
this mighty scourge of war may speedily pass away. Yet, if God wills that it continue until all
the wealth piled by the bondsman's two hundred and fifty years of unrequited toil shall be sunk,
and until every drop of blood drawn with the lash shall be paid by another drawn with the sword,
as was said three thousand years ago, so still it must be said "the judgments of the Lord are
true and righteous altogether."

With malice toward none, with charity for all, with firmness in the right as God gives us to see
the right, let us strive on to finish the work we are in, to bind up the nation's wounds, to
care for him who shall have borne the battle and for his widow and his orphan, to do all which
may achieve and cherish a just and lasting peace among ourselves and with all nations.
"""

# The last paragraph is held out; training never sees it.
HELD_OUT_PARAGRAPHS: Final[int] = 1

Pair = tuple[int, int]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE PAIN: Byte-Pair Encoding with Plain Lists
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def split_corpus(text: str = CORPUS, held_out: int = HELD_OUT_PARAGRAPHS) -> tuple[str, str]:
    r"""
    Split ``text`` into training and held-out parts at paragraph boundaries.

    The last ``held_out`` blank-line-separated paragraphs are held out. Line
    breaks inside a paragraph become single spaces.

    Examples
    --------
    >>> split_corpus("a\nb\n\nc\n\nd", held_out=1)
    ('a b\n\nc', 'd')
    >>> train_text, held_text = split_corpus()
    >>> held_text.startswith("With malice toward none")
    True
    >>> "malice" in train_text
    False
    """
    paragraphs = [" ".join(p.split()) for p in text.strip().split("\n\n")]
    return "\n\n".join(paragraphs[:-held_out]), "\n\n".join(paragraphs[-held_out:])


def count_pairs(ids: Sequence[int]) -> Counter[Pair]:
    """
    Count each adjacent pair of ids, overlaps included.

    Examples
    --------
    >>> count_pairs([1, 2, 1, 2, 1])
    Counter({(1, 2): 2, (2, 1): 2})
    >>> count_pairs([7])
    Counter()

    Complexity
    ----------
    Time
        O(n) for n ids.
    Space
        O(k) for k distinct pairs.
    """
    return Counter(pairwise(ids))


def merge_pair(ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """
    Replace each non-overlapping occurrence of ``pair`` with ``new_id``, left to right.

    Examples
    --------
    >>> merge_pair([1, 2, 3, 1, 2], (1, 2), 9)
    [9, 3, 9]

    Overlapping occurrences merge from the left, so three equal ids leave one behind:

    >>> merge_pair([5, 5, 5], (5, 5), 9)
    [9, 5]

    Complexity
    ----------
    Time
        O(n) for n ids.
    Space
        O(n).
    """
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(text: str, num_merges: int) -> list[Pair]:
    """
    Learn up to ``num_merges`` merges from the UTF-8 bytes of ``text``.

    Merge ``i`` creates token ``256 + i`` from its pair of earlier tokens. Each
    round merges the most frequent pair; a tie goes to the pair with the smaller
    ``(left, right)`` ids, so training is deterministic. Training stops early
    when no pair occurs twice, because merging a pair seen once saves nothing.

    Examples
    --------
    >>> train("aaabdaaabac", 3)
    [(97, 97), (97, 98), (256, 257)]

    After the first merge, ``(256, 97)`` and ``(97, 98)`` each occur twice, so
    the second merge is ``(97, 98)``: the smaller ids win the tie. A pair that
    occurs once is never merged:

    >>> train("abcd", 5)
    []

    Complexity
    ----------
    Time
        O(n·m) for n bytes and m merges: each merge recounts every pair.
    Space
        O(n).
    """
    ids = list(text.encode("utf-8"))
    merges: list[Pair] = []
    for _ in range(num_merges):
        counts = count_pairs(ids)
        if not counts:
            break
        best, seen = max(counts.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
        if seen < 2:
            break
        ids = merge_pair(ids, best, 256 + len(merges))
        merges.append(best)
    return merges


def encode(text: str, merges: Sequence[Pair]) -> list[int]:
    """
    Encode ``text`` by applying ``merges`` in the order they were learned.

    Each round merges the pair with the lowest rank (earliest learned) that is
    present, until no learned pair remains. Any string encodes: bytes no merge
    touches stay as ids below 256.

    Examples
    --------
    >>> merges = train("aaabdaaabac", 3)
    >>> encode("aaab", merges)
    [258]
    >>> encode("", merges)
    []

    Rank decides overlapping pairs: ``ab`` was learned before ``bc``, so it wins
    the shared ``b``:

    >>> encode("abc", [(97, 98), (98, 99)])
    [256, 99]

    The deliberate break: replay the same merges from last to first. Later merges
    need tokens that earlier merges build, so they never fire, and the text stays
    longer. The ids still decode, but the compression is lost:

    >>> wrong = list(b"aaab")
    >>> for rank in reversed(range(len(merges))):
    ...     wrong = merge_pair(wrong, merges[rank], 256 + rank)
    >>> wrong
    [256, 257]
    >>> decode(wrong, merges) == b"aaab"
    True
    >>> len(wrong) > len(encode("aaab", merges))
    True

    Complexity
    ----------
    Time
        O(n·m) worst case for n bytes and m merges.
    Space
        O(n).
    """
    rank = {pair: i for i, pair in enumerate(merges)}
    ids = list(text.encode("utf-8"))
    while True:
        known = [pair for pair in count_pairs(ids) if pair in rank]
        if not known:
            return ids
        best = min(known, key=rank.__getitem__)
        ids = merge_pair(ids, best, 256 + rank[best])


def build_vocab(merges: Sequence[Pair]) -> dict[int, bytes]:
    """
    Map every token id to the bytes it stands for.

    Ids 0 to 255 are single bytes; token ``256 + i`` is the concatenation of
    the bytes of the two tokens in ``merges[i]``.

    Examples
    --------
    >>> vocab = build_vocab(train("aaabdaaabac", 3))
    >>> len(vocab)
    259
    >>> vocab[97], vocab[256], vocab[258]
    (b'a', b'aa', b'aaab')
    """
    vocab = {i: bytes([i]) for i in range(256)}
    for i, (left, right) in enumerate(merges):
        vocab[256 + i] = vocab[left] + vocab[right]
    return vocab


def decode(ids: Sequence[int], merges: Sequence[Pair]) -> bytes:
    """
    Return the bytes ``ids`` stand for under ``merges``.

    Raises ``KeyError`` for an id outside the vocabulary. Decode the result with
    ``.decode("utf-8")`` to get text; ids cut in the middle of a character are not
    valid UTF-8 on their own.

    Examples
    --------
    >>> merges = train("aaabdaaabac", 3)
    >>> decode([258, 100, 258, 97, 99], merges)
    b'aaabdaaabac'
    >>> decode([], merges)
    b''

    The deliberate break: a decoder that ignores the merges only knows the 256
    bytes, so a merged token has no meaning and the round trip fails:

    >>> decode([258], [])
    Traceback (most recent call last):
        ...
    KeyError: 258

    Complexity
    ----------
    Time
        O(m + n·L) for m merges, n ids, and longest token L bytes.
    Space
        O(m·L + n·L).
    """
    vocab = build_vocab(merges)
    return b"".join(vocab[i] for i in ids)


def bytes_per_token(text: str, merges: Sequence[Pair]) -> float:
    """
    Return the UTF-8 bytes of ``text`` divided by its token count under ``merges``.

    With no merges every byte is a token, so the ratio is 1.0. Empty text has no
    tokens and returns 0.0.

    Examples
    --------
    >>> bytes_per_token("aaabdaaabac", [])
    1.0
    >>> bytes_per_token("aaabdaaabac", train("aaabdaaabac", 3))
    2.2
    >>> bytes_per_token("", [])
    0.0
    """
    ids = encode(text, merges)
    return len(text.encode("utf-8")) / len(ids) if ids else 0.0


def compression_table(
    text: str, merges: Sequence[Pair], steps: Sequence[int]
) -> list[tuple[int, int, float]]:
    """
    Return ``(merges used, tokens, bytes per token)`` on ``text`` for each step.

    A step larger than ``len(merges)`` uses every merge available.

    Examples
    --------
    >>> merges = train("aaabdaaabac", 3)
    >>> compression_table("aaabdaaabac", merges, [0, 1, 3])
    [(0, 11, 1.0), (1, 9, 1.2222222222222223), (3, 5, 2.2)]
    >>> compression_table("aaab", merges, [99])
    [(99, 1, 4.0)]
    """
    rows = []
    for step in steps:
        ids = encode(text, merges[:step])
        rows.append((step, len(ids), len(text.encode("utf-8")) / len(ids)))
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI ARGUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class Args:
    """Parsed command-line arguments.

    Attributes
    ----------
    doctest : bool
        Run the doctests before the demo. On by default; ``--no-doctest``
        turns it off.
    doctest_only : bool
        Run the doctests and exit without running the demo.
    verbose : bool
        Report each doctest rather than only failures.
    show_index : bool
        Print the lesson index instead of running anything.
    """

    doctest: bool
    doctest_only: bool
    verbose: bool
    show_index: bool


def parse_args(argv: list[str] | None = None) -> Args:
    """
    Parse the lesson's command-line flags.

    Examples
    --------
    >>> parse_args(["--doctest-only"]).doctest_only
    True
    >>> parse_args(["--no-doctest"]).doctest
    False
    """
    parser = argparse.ArgumentParser(description=f"Run {LESSON_ID}")
    parser.add_argument(
        "--doctest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run doctests before demo (default: True)",
    )
    parser.add_argument("--doctest-only", action="store_true", help="Run doctests and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose doctest output")
    parser.add_argument(
        "--index",
        action="store_true",
        dest="show_index",
        help="Show lesson index (dependencies, successors, related)",
    )
    ns = parser.parse_args(argv)
    return Args(
        doctest=ns.doctest,
        doctest_only=ns.doctest_only,
        verbose=ns.verbose,
        show_index=ns.show_index,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX DISPLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _format_mapping(name: str, items: Mapping[str, str]) -> str:
    """
    Format a ``lesson_id -> reason`` mapping as an indented list.

    Examples
    --------
    >>> print(_format_mapping("Related", {}))
    Related: (none)
    >>> print(_format_mapping("Next", {"tier_9_x": "why"}))
    Next:
       → tier_9_x: why
    """
    if not items:
        return f"{name}: (none)"
    lines = "\n".join(f"   → {lesson_id}: {reason}" for lesson_id, reason in items.items())
    return f"{name}:\n{lines}"


def show_lesson_index() -> None:
    """
    Print where this lesson sits in the curriculum.

    Examples
    --------
    >>> show_lesson_index()  # doctest: +ELLIPSIS
    ═...
      LESSON INDEX: tier_6_lesson_2_toy_bpe
    ...
    Prerequisites (hard deps):
       → tier_6_lesson_1_naive_tokenization: Regex tokenizers and where they break
    ...
       → Power-up in: tier_6_lesson_3_tiktoken
    """
    print("═" * 70)
    print(f"  LESSON INDEX: {LESSON_ID}")
    print("═" * 70)
    print()
    print(_format_mapping("Prerequisites (hard deps)", PREREQUISITES))
    print()
    print(_format_mapping("Dependencies (soft deps)", DEPENDENCIES))
    print()
    print(_format_mapping("Successors (unlocks)", SUCCESSORS))
    print()
    print(_format_mapping("Dependents (rely on this)", DEPENDENTS))
    print()
    print(_format_mapping("Related (optional)", RELATED))
    print()
    print("Power-Up Timeline:")
    if PAIN_FROM_LESSON:
        print(f"   ← Pain from: {PAIN_FROM_LESSON}")
    if POWER_UP_AVAILABLE:
        print("   ✓ Power-up: Available in THIS lesson")
    elif POWER_UP_IN_LESSON:
        print(f"   → Power-up in: {POWER_UP_IN_LESSON}")
    else:
        print("   → Power-up in: (future lesson TBD)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEMONSTRATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def demo(num_merges: int = 200, steps: Sequence[int] = (0, 10, 50, 100, 200)) -> None:
    """
    Train on the embedded corpus, then print merges, compression, and round trips.

    Examples
    --------
    >>> demo(num_merges=20, steps=(0, 20))  # doctest: +ELLIPSIS
    ═...
    First merges (token id: bytes):
    ...
    Round trip on held-out text: True
    ...
    """
    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    print()
    train_text, held_text = split_corpus()
    merges = train(train_text, num_merges)
    print(
        f"Trained {len(merges)} merges on {len(train_text.encode())} bytes; "
        f"held out {len(held_text.encode())} bytes"
    )
    print()
    vocab = build_vocab(merges)
    print("First merges (token id: bytes):")
    for i in range(min(10, len(merges))):
        print(f"  {256 + i}: {vocab[256 + i]!r}")
    print()
    print(f"{'merges':<10}{'tokens':<10}bytes per token (held-out)")
    for used, tokens, ratio in compression_table(held_text, merges, steps):
        print(f"{used:<10}{tokens:<10}{ratio:.3f}")
    print()
    ids = encode(held_text, merges)
    print(f"Round trip on held-out text: {decode(ids, merges).decode() == held_text}")
    odd = "naïve café 世界 🙂"
    odd_ids = encode(odd, merges)
    raw = sum(1 for i in odd_ids if i < 256)
    print(f"Unseen characters: {odd!r}")
    print(
        f"  {len(odd.encode())} bytes -> {len(odd_ids)} tokens, {raw} of them raw bytes; "
        f"round trip: {decode(odd_ids, merges).decode() == odd}"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_doctests(module: ModuleType | None = None, verbose: bool = False) -> doctest.TestResults:
    r"""
    Run a module's doctests with the option flags pytest uses.

    ``pyproject.toml`` gives pytest ``ELLIPSIS`` and ``NORMALIZE_WHITESPACE``;
    passing the same flags here keeps an example's verdict the same under both.

    Parameters
    ----------
    module : ModuleType or None
        Module to test; ``None`` tests ``__main__``, as ``doctest.testmod`` does.
    verbose : bool
        Report each example rather than only failures.

    Returns
    -------
    doctest.TestResults
        Counts of failed and attempted examples.

    Examples
    --------
    Output that differs only in spacing, or is elided with ``...``, passes:

    >>> import types
    >>> doc = ">>> print('a  b')\na b\n>>> print('abc')\na...\n"
    >>> results = run_doctests(types.ModuleType("flags_demo", doc))
    >>> results.failed, results.attempted
    (0, 2)
    """
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    return doctest.testmod(module, verbose=verbose, optionflags=flags)


def main(argv: list[str] | None = None) -> int:
    """
    Parse arguments, run the doctests unless told not to, then run the demo.

    Returns
    -------
    int
        Exit code: 0 on success, 1 when a doctest fails.

    Examples
    --------
    >>> main(["--index"])  # doctest: +ELLIPSIS
    ═...
    0
    """
    args = parse_args(argv)
    if args.show_index:
        show_lesson_index()
        return 0
    if args.doctest:
        results = run_doctests(verbose=args.verbose)
        if results.failed > 0:
            print(f"DOCTESTS FAILED: {results.failed}/{results.attempted}")
            return 1
        print(f"All {results.attempted} doctests passed\n")
    if args.doctest_only:
        return 0
    demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
