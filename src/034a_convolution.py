#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = ["numpy>=2.5.4"]
# ///
"""
Lesson 3.4a: Convolution as a Sliding Dot Product (side quest).

You slide a small grid of weights, a kernel, across an image and take a dot
product at every position. The grid of results is a feature map: bright where
the image looks like the kernel. With hand-made line detectors at four
orientations, each map lights up for lines at its own angle, which is the first
thing vision networks learn and the starting point of Distill's Circuits work.

Source: Olah et al., "An Overview of Early Vision in InceptionV1", Distill
(2020), https://distill.pub/2020/circuits/early-vision/ , and Schubert et al.,
"High-Low Frequency Detectors", Distill (2021),
https://distill.pub/2020/circuits/frequency-edges/

Prerequisites (must complete first)
----------------------------------
- Lesson 3.1: Dot Product and Cosine Similarity (the operation at each position)
- Lesson 3.4: NumPy Upgrade (vectorization and matmul)

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. Compute a "valid" 2-D convolution with nested loops and with NumPy.
2. Check two implementations against each other, and see the check fail
   when one of them is wrong.
3. Build oriented line detectors and read their responses as a tuning table.

The Big Idea
------------
A convolution reuses one small set of weights at every position, so the same
pattern detector scans the whole image; the feature map records where it found
its pattern.

Why It Matters
--------------
Every convolutional network, from LeNet to the InceptionV1 that Distill's
Circuits thread dissects, stacks convolutions like these. Its first layer learns
oriented edge and line detectors much like the hand-made kernels here, so
reading a feature map is the first step in reading a vision model.

Stdlib Pain → Library Power-Up
------------------------------
Pain (this lesson):
- Four nested loops: output row, output column, kernel row, kernel column.
- Lines of code: ~8.
- Runtime: tens of milliseconds for a 64x64 image and a 5x5 kernel.
- Common bugs: off-by-one output sizes, and mixing up rows and columns of the
  kernel, which silently transposes it.

Power-Up (this lesson — NumPy):
- ``np.lib.stride_tricks.sliding_window_view`` exposes every kxk window as a
  view without copying, and ``np.einsum`` takes all the dot products at once.
- Lines of code: ~2.
- Runtime: well under a millisecond for the same image.
- Key APIs: ``sliding_window_view``, ``np.einsum``.

Complexity Analysis
-------------------
Pain (manual), for an HxW image and a kxk kernel:
- Time:  O((H-k+1)·(W-k+1)·k²).
- Space: O((H-k+1)·(W-k+1)) for the feature map.

Power-Up:
- Time:  the same count of multiply-adds, run in compiled code.
- Space: the same feature map; the windows are a view, not a copy.

Execution
---------
Project-style:
    uv run python src/034a_convolution.py

Script-style (self-contained deps):
    uv run --script src/034a_convolution.py

Options:
    python src/034a_convolution.py --doctest-only
    python src/034a_convolution.py --no-doctest
    python src/034a_convolution.py --index
    python src/034a_convolution.py -v

Quality Gates
-------------
    uv run pytest src/034a_convolution.py
    uv run ruff check src/034a_convolution.py
    uv run ruff format --check src/034a_convolution.py
    uv run mypy --strict src/034a_convolution.py

Module Doctests
---------------
The kernel [[1, 0], [0, -1]] subtracts each pixel's lower-right neighbour, which
is 4 more everywhere in a 3x3 grid counting 1 to 9:

>>> conv2d_pain([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 0], [0, -1]])
[[-4.0, -4.0], [-4.0, -4.0]]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    type Array = NDArray[np.float64]
    type ConvFn = Callable[[Array, Array], Array]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON_ID: Final[str] = "tier_3_lesson_4a_convolution"

# This lesson introduces its own power-up: NumPy's sliding windows.
POWER_UP_AVAILABLE: Final[bool] = True

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason

PREREQUISITES: Final[Mapping[str, str]] = {
    "tier_3_lesson_1_dot_product": "The operation taken at every position",
    "tier_3_lesson_4_numpy": "Vectorized arrays and einsum",
}

DEPENDENCIES: Final[Mapping[str, str]] = {}

SUCCESSORS: Final[Mapping[str, str]] = {
    "tier_5_lesson_2c_tiny_cnn": "Learns its kernels instead of hand-making them",
}

DEPENDENTS: Final[Mapping[str, str]] = {
    "tier_5_lesson_2d_reading_a_unit": "Reads a trained network's units with tuning curves",
}

RELATED: Final[Mapping[str, str]] = {
    "tier_5_lesson_2a_superposition": "Another way features share a small space",
}

POWER_UP_IN_LESSON: Final[str | None] = None
PAIN_FROM_LESSON: Final[str | None] = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE PAIN: Convolution With Nested Loops (stdlib)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def conv2d_pain(
    image: Sequence[Sequence[float]], kernel: Sequence[Sequence[float]]
) -> list[list[float]]:
    """
    Slide ``kernel`` over ``image`` and take the dot product at each position.

    "Valid" mode: the kernel never hangs off the edge, so an HxW image and a
    kxk kernel give an (H-k+1)x(W-k+1) map. Like deep-learning libraries, this
    does not flip the kernel (strictly, it is cross-correlation).

    Examples
    --------
    The kernel [[0, 1], [0, 0]] picks each window's top-right pixel:

    >>> conv2d_pain([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0, 1], [0, 0]])
    [[2.0, 3.0], [5.0, 6.0]]
    >>> conv2d_pain([[1, 2], [3, 4]], [[1, 1], [1, 1]])
    [[10.0]]
    """
    kh, kw = len(kernel), len(kernel[0])
    out = []
    for i in range(len(image) - kh + 1):
        row = []
        for j in range(len(image[0]) - kw + 1):
            total = 0.0
            for a in range(kh):
                for b in range(kw):
                    total += image[i + a][j + b] * kernel[a][b]
            row.append(total)
        out.append(row)
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE POWER-UP: Sliding Windows (NumPy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def conv2d(image: Array, kernel: Array) -> Array:
    """
    Compute the same valid convolution as ``conv2d_pain`` with NumPy windows.

    Examples
    --------
    >>> import numpy as np
    >>> conv2d(np.arange(1.0, 10.0).reshape(3, 3), np.array([[1.0, 0.0], [0.0, -1.0]])).tolist()
    [[-4.0, -4.0], [-4.0, -4.0]]
    >>> conv2d(np.zeros((16, 16)), np.zeros((5, 5))).shape
    (12, 12)
    """
    windows = np.lib.stride_tricks.sliding_window_view(image, kernel.shape)
    return np.asarray(np.einsum("ijkl,kl->ij", windows, kernel), dtype=np.float64)


def parity_cases() -> list[tuple[Array, Array]]:
    """
    Return small image-kernel pairs, including asymmetric kernels that catch a transposed kernel.

    Examples
    --------
    >>> len(parity_cases()) >= 3
    True
    """
    rng = np.random.default_rng(0)
    return [
        (np.arange(1.0, 10.0).reshape(3, 3), np.array([[1.0, 0.0], [0.0, -1.0]])),
        (rng.normal(size=(6, 7)), rng.normal(size=(3, 3))),
        (rng.normal(size=(9, 9)), rng.normal(size=(4, 4))),
    ]


def check_parity(impl: ConvFn, cases: Sequence[tuple[Array, Array]] | None = None) -> None:
    """
    Raise ``AssertionError`` unless ``impl`` matches ``conv2d_pain`` on every case.

    ``cases`` defaults to ``parity_cases()``.

    Examples
    --------
    The NumPy version agrees with the loops:

    >>> check_parity(conv2d)

    The deliberate break: a version that transposes the kernel, the classic
    row-column mix-up, fails the check:

    >>> check_parity(lambda image, kernel: conv2d(image, kernel.T))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    AssertionError: parity failed on case ...
    """
    for i, (image, kernel) in enumerate(cases if cases is not None else parity_cases()):
        want = np.array(conv2d_pain(image.tolist(), kernel.tolist()))
        got = impl(image, kernel)
        if got.shape != want.shape or not np.allclose(got, want):
            msg = f"parity failed on case {i}"
            raise AssertionError(msg)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ORIENTED LINES AND LINE DETECTORS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def distance_to_line(size: int, angle: float) -> Array:
    """
    Return each pixel's distance from the line through the grid's center at ``angle`` degrees.

    Angle 0 is a horizontal line and 90 a vertical one.

    Examples
    --------
    >>> d = distance_to_line(5, 0.0)
    >>> d[2].tolist(), d[0, 2].item()
    ([0.0, 0.0, 0.0, 0.0, 0.0], 2.0)
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    center = (size - 1) / 2
    theta = np.deg2rad(angle)
    return np.asarray(
        np.abs(-(xx - center) * np.sin(theta) + (yy - center) * np.cos(theta)), dtype=np.float64
    )


def line_image(angle: float, size: int = 16) -> Array:
    """
    Draw a soft line through the center at ``angle`` degrees, brightness in [0, 1].

    Examples
    --------
    >>> img = line_image(0.0)
    >>> img.shape, float(img.max()) <= 1.0
    ((16, 16), True)
    >>> int(img.sum(axis=1).argmax()) in (7, 8)
    True
    """
    return np.exp(-(distance_to_line(size, angle) ** 2))


def line_kernel(angle: float, size: int = 5) -> Array:
    """
    Build a zero-mean detector for lines at ``angle`` degrees.

    The kernel is a soft line minus its mean, so a flat patch gives 0, the
    matching line gives a large positive response, and a crossing line a small one.

    Examples
    --------
    >>> k = line_kernel(45.0)
    >>> k.shape, abs(float(k.sum())) < 1e-9
    ((5, 5), True)
    """
    k = np.exp(-((distance_to_line(size, angle) / 0.8) ** 2))
    return k - k.mean()


def response(image: Array, kernel: Array) -> float:
    """
    Return the mean of the rectified feature map, max(0, conv2d(image, kernel)).

    Examples
    --------
    >>> response(line_image(0.0), line_kernel(0.0)) > response(line_image(0.0), line_kernel(90.0))
    True
    """
    return float(np.maximum(conv2d(image, kernel), 0.0).mean())


def tuning(angles: Sequence[float]) -> Array:
    """
    Return the response of each angle's kernel (rows) to each angle's line (columns).

    Examples
    --------
    Each detector responds most to the line at its own angle:

    >>> table = tuning((0.0, 45.0, 90.0, 135.0))
    >>> [int(row.argmax()) for row in table]
    [0, 1, 2, 3]
    """
    return np.array([[response(line_image(b), line_kernel(a)) for b in angles] for a in angles])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEEING IT: ASCII Feature Maps (stdlib)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAMP: Final[str] = " .:-=+*#%@"


def shade(feature_map: Array, top: float | None = None) -> str:
    """
    Draw a map as text, one doubled character per value, darker for larger values.

    Negative values count as 0. ``top`` is the value drawn darkest; it defaults
    to the map's own maximum, and passing one shared ``top`` puts two maps on
    the same scale.

    Examples
    --------
    >>> import numpy as np
    >>> shade(np.array([[0.0, 1.0], [0.5, -2.0]])).splitlines()
    ['  @@', '++  ']
    >>> shade(np.array([[0.0, 1.0], [0.5, -2.0]]), top=2.0).splitlines()
    ['  ++', '::  ']
    """
    values = np.maximum(feature_map, 0.0)
    if top is None:
        top = float(values.max())
    rows = []
    for row in values:
        cells = (
            RAMP[min(int(v / top * len(RAMP)), len(RAMP) - 1)] if top > 0 else " " for v in row
        )
        rows.append("".join(c * 2 for c in cells))
    return "\n".join(rows)


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
      LESSON INDEX: tier_3_lesson_4a_convolution
    ...
    Prerequisites (hard deps):
       → tier_3_lesson_1_dot_product: The operation taken at every position
    ...
       ✓ Power-up: Available in THIS lesson
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


def demo(angles: Sequence[float] = (0.0, 45.0, 90.0, 135.0)) -> None:
    r"""
    Print the tuning table, two feature maps on one scale, and the stdlib against NumPy timing.

    Examples
    --------
    >>> demo()  # doctest: +ELLIPSIS
    ═...
    kernel \ line    0        45       90       135
    0 deg  ...
    Feature map of the 45 deg detector on the 45 deg line:
    ...
    64x64 image, 5x5 kernel: stdlib ... ms, NumPy ... ms
    """
    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    print()
    table = tuning(angles)
    header = f"{'kernel \\ line':<17}" + "".join(f"{a:<9g}" for a in angles)
    print(header.rstrip())
    for a, row in zip(angles, table, strict=True):
        print((f"{f'{a:g} deg':<17}" + "".join(f"{v:<9.3f}" for v in row)).rstrip())
    maps = {line: conv2d(line_image(line), line_kernel(45.0)) for line in (45.0, 135.0)}
    scale = float(maps[45.0].max())
    for line, feature_map in maps.items():
        print()
        print(f"Feature map of the 45 deg detector on the {line:g} deg line:")
        print(shade(feature_map, top=scale))
    rng = np.random.default_rng(0)
    image, kernel = rng.random((64, 64)), line_kernel(45.0)
    start = time.perf_counter()
    conv2d_pain(image.tolist(), kernel.tolist())
    pain_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    conv2d(image, kernel)
    numpy_ms = (time.perf_counter() - start) * 1000
    print()
    print(f"64x64 image, 5x5 kernel: stdlib {pain_ms:.1f} ms, NumPy {numpy_ms:.2f} ms")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


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
    import doctest

    args = parse_args(argv)
    if args.show_index:
        show_lesson_index()
        return 0
    if args.doctest:
        results = doctest.testmod(verbose=args.verbose)
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
