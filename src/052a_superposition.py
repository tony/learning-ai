#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = ["numpy>=2.5.4"]
# ///
"""
Lesson 5.2a: Toy Models of Superposition (side quest).

You train a tiny autoencoder that squeezes n sparse features through m < n
hidden dimensions, with every gradient derived by hand, and count how many
features it manages to store. When the inputs are dense it stores exactly m
features and drops the rest. When the inputs are sparse it stores all n as
overlapping directions, accepting a little interference to keep every feature.

Source: Elhage et al., "Toy Models of Superposition", Transformer Circuits
Thread (2022), https://transformer-circuits.pub/2022/toy_model/index.html

Prerequisites (must complete first)
----------------------------------
- Lesson 5.1: Manual Backprop (deriving gradients by the chain rule)
- Lesson 4.1: Finite Differences (checking a gradient numerically)

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. Derive the gradient of a ReLU autoencoder with tied weights by hand.
2. Check that gradient against central differences, and see the check fail
   when a term is missing.
3. Show that sparsity decides how many features fit into fewer dimensions.

The Big Idea
------------
A network can store more features than it has dimensions by letting their
directions overlap, which costs interference whenever two features are active
together. Sparse features rarely are, so the overlap is cheap and the network
takes it.

Why It Matters
--------------
Superposition is one proposed reason that single neurons in real networks
respond to several unrelated things (polysemanticity). Nick Cammarata's 2026
Goodfire essay "Neural Networks Are Already Clean"
(https://already-clean.goodfire.com/) accepts the toy's results but argues
that real networks' polysemantic neurons have other causes, so the toy shows
what superposition is, not proof that real networks rely on it.

Stdlib Pain → Library Power-Up
------------------------------
Pain (this lesson):
- You derive the gradient on paper, then write it as five lines of NumPy; a
  dropped term or a missing ReLU mask trains silently to a worse answer.
- Lines of code: ~10 for the gradient, ~15 for Adam.
- Runtime: about a minute for the sparsity sweep (five runs of 10,000 steps).
- Common bugs: omitting the encoder term ``W D^T X`` that appears because the
  same ``W`` encodes and decodes.

Power-Up (Lesson 5.3 — PyTorch):
- ``loss.backward()`` produces both gradient terms from the forward pass, and
  ``torch.optim.Adam`` replaces the hand-written update.
- Key APIs: ``torch.autograd``, ``torch.optim.Adam``.

Complexity Analysis
-------------------
Pain (manual), for batch B, n features, m hidden dimensions, k steps:
- Time:  O(k·B·n·m).
- Space: O(B·n + n·m).

Execution
---------
Project-style:
    uv run python src/052a_superposition.py

Script-style (self-contained deps):
    uv run --script src/052a_superposition.py

On a busy machine, NumPy's BLAS threads compete with other work and can stretch
the demo from about a minute to several; one thread avoids that, since the
matrices here are small:
    OPENBLAS_NUM_THREADS=1 uv run python src/052a_superposition.py

Options:
    python src/052a_superposition.py --doctest-only
    python src/052a_superposition.py --no-doctest
    python src/052a_superposition.py --index
    python src/052a_superposition.py -v

Quality Gates
-------------
    uv run pytest src/052a_superposition.py
    uv run ruff check src/052a_superposition.py
    uv run ruff format --check src/052a_superposition.py
    uv run mypy --strict src/052a_superposition.py

Module Doctests
---------------
With W = 2, b = 0, and one input x = 1, the loss is (1 - 4)² = 9 and
d/dW of (W² - 1)² = 4W(W² - 1) is 24:

>>> import numpy as np
>>> W, b, X = np.array([[2.0]]), np.zeros(1), np.array([[1.0]])
>>> loss, grad_W, grad_b = loss_and_grads(W, b, X, importance(1))
>>> float(loss), float(grad_W[0, 0])
(9.0, 24.0)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    type Array = NDArray[np.float64]
    type GradFn = Callable[[Array, Array, Array, Array], tuple[float, Array, Array]]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON_ID: Final[str] = "tier_5_lesson_2a_superposition"

# Autograd and torch.optim.Adam arrive with PyTorch.
POWER_UP_AVAILABLE: Final[bool] = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason

PREREQUISITES: Final[Mapping[str, str]] = {
    "tier_5_lesson_1_manual_backprop": "Hand-derived gradients",
    "tier_4_lesson_1_finite_differences": "Numeric gradients to check against",
}

DEPENDENCIES: Final[Mapping[str, str]] = {
    "tier_3_lesson_4_numpy": "Vectorized matmul and broadcasting",
}

SUCCESSORS: Final[Mapping[str, str]] = {
    "tier_5_lesson_3_pytorch": "Autograd replaces the hand-written gradient",
}

DEPENDENTS: Final[Mapping[str, str]] = {
    "tier_8_lesson_2a_grokking": "Another small model whose internals you can inspect",
}

RELATED: Final[Mapping[str, str]] = {
    "tier_4_lesson_2a_momentum": "Adam builds on momentum",
    "tier_8_lesson_2a_grokking": "Both read structure out of a small trained network",
}

POWER_UP_IN_LESSON: Final[str | None] = "tier_5_lesson_3_pytorch"
PAIN_FROM_LESSON: Final[str | None] = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE PAIN: A ReLU Autoencoder With Hand-Derived Gradients
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def importance(n: int) -> Array:
    """
    Return the weight 0.9**i of each feature's reconstruction error.

    Examples
    --------
    >>> importance(3).round(2).tolist()
    [1.0, 0.9, 0.81]
    """
    return np.asarray(0.9 ** np.arange(n), dtype=np.float64)


def sample_features(rng: np.random.Generator, batch: int, n: int, sparsity: float) -> Array:
    """
    Draw a ``(batch, n)`` matrix of features in [0, 1], each zeroed with probability ``sparsity``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> sample_features(rng, 4, 3, 0.5).shape
    (4, 3)
    >>> bool((sample_features(rng, 100, 3, 1.0) == 0).all())
    True
    >>> bool((sample_features(rng, 100, 3, 0.0) > 0).all())
    True
    """
    values = rng.random((batch, n))
    return np.asarray(values * (rng.random((batch, n)) >= sparsity), dtype=np.float64)


def loss_and_grads(W: Array, b: Array, X: Array, imp: Array) -> tuple[float, Array, Array]:
    """
    Return the loss and its gradients with respect to ``W`` and ``b``.

    The forward pass is H = X Wᵀ, Z = H W + b, X' = ReLU(Z), and the loss is
    the batch mean of Σᵢ impᵢ (Xᵢ - X'ᵢ)². Backward, with D = ∂loss/∂Z::

        D = 2 · imp · (X' - X) · (Z > 0) / B
        grad_W = Hᵀ D + W Dᵀ X      (decoder term + encoder term)
        grad_b = Σ over the batch of D

    ``W`` appears twice, so its gradient has two terms.

    Examples
    --------
    One feature, one dimension, W = 2: the loss is (1 - 4)² = 9, and
    d/dW of (W² - 1)² = 4W(W² - 1) is 24, the sum of two equal terms of 12:

    >>> import numpy as np
    >>> W, b, X = np.array([[2.0]]), np.zeros(1), np.array([[1.0]])
    >>> loss, grad_W, grad_b = loss_and_grads(W, b, X, importance(1))
    >>> float(loss), float(grad_W[0, 0]), float(grad_b[0])
    (9.0, 24.0, 6.0)

    With W = 0 and b = 1 the output is 1 for an input of 0, so the loss is 1
    and only the bias gradient is nonzero:

    >>> W, b, X = np.zeros((1, 1)), np.ones(1), np.zeros((1, 1))
    >>> loss, grad_W, grad_b = loss_and_grads(W, b, X, importance(1))
    >>> float(loss), float(grad_W[0, 0]), float(grad_b[0])
    (1.0, 0.0, 2.0)

    With W = 2 and b = -5 the pre-activation is 2·2 - 5 = -1, so ReLU is
    inactive: the loss is 1 but the gradients are exactly zero.

    >>> W, b, X = np.array([[2.0]]), np.full(1, -5.0), np.array([[1.0]])
    >>> loss, grad_W, grad_b = loss_and_grads(W, b, X, importance(1))
    >>> float(loss), float(grad_W[0, 0]), float(grad_b[0])
    (1.0, 0.0, 0.0)
    """
    H = X @ W.T
    Z = H @ W + b
    out = np.maximum(Z, 0.0)
    loss = float(np.mean(np.sum(imp * (X - out) ** 2, axis=1)))
    D = 2.0 * imp * (out - X) * (Z > 0) / len(X)
    return loss, H.T @ D + W @ D.T @ X, D.sum(axis=0)


def gradient_error(
    grad_fn: GradFn,
    W: Array,
    b: Array,
    X: Array,
    imp: Array,
    eps: float = 1e-5,
) -> float:
    """
    Return the largest gap between ``grad_fn``'s gradient for ``W`` and central differences.

    ``grad_fn`` has the signature of ``loss_and_grads``. Each entry of ``W`` is
    nudged by ±``eps`` and the loss from ``grad_fn`` is differenced. An exact
    gradient gives an error near ``eps``², a wrong one gives an error of the
    size of the missing term.

    Examples
    --------
    The hand-derived gradient agrees with the numeric one:

    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> W = rng.normal(size=(2, 5))
    >>> b = rng.normal(size=5)
    >>> X = sample_features(rng, 8, 5, 0.3)
    >>> gradient_error(loss_and_grads, W, b, X, importance(5)) < 1e-6
    True

    The deliberate break: a gradient missing the encoder term W Dᵀ X is far off.

    >>> def broken(W, b, X, imp):
    ...     H = X @ W.T
    ...     Z = H @ W + b
    ...     D = 2 * imp * (np.maximum(Z, 0) - X) * (Z > 0) / len(X)
    ...     loss, _, _ = loss_and_grads(W, b, X, imp)
    ...     return loss, H.T @ D, D.sum(axis=0)
    >>> gradient_error(broken, W, b, X, importance(5)) > 0.1
    True
    """
    _, analytic, _ = grad_fn(W, b, X, imp)
    numeric = np.zeros_like(W)
    for index in np.ndindex(*W.shape):
        up, down = W.copy(), W.copy()
        up[index] += eps
        down[index] -= eps
        numeric[index] = (grad_fn(up, b, X, imp)[0] - grad_fn(down, b, X, imp)[0]) / (2 * eps)
    return float(np.max(np.abs(analytic - numeric)))


def train(
    n: int,
    m: int,
    sparsity: float,
    steps: int = 10_000,
    seed: int = 0,
    batch: int = 1024,
    lr: float = 1e-3,
) -> tuple[Array, Array]:
    """
    Train the autoencoder with hand-written Adam and return ``(W, b)``.

    Adam uses betas 0.9 and 0.999 and eps 1e-8. The seeded generator draws the
    initial ``W`` and then a fresh batch per step, so a run repeats exactly.

    Examples
    --------
    The paper's small case, n = 5 features in m = 2 dimensions. Dense inputs
    fit only two features; the rest of ``W`` shrinks away:

    >>> W, b = train(5, 2, sparsity=0.0, steps=2000, lr=0.01)
    >>> W.shape, b.shape
    ((2, 5), (5,))
    >>> features_represented(W)
    2

    Sparse inputs let all five share the two dimensions:

    >>> W, b = train(5, 2, sparsity=0.95, steps=2000, lr=0.01)
    >>> features_represented(W)
    5
    """
    rng = np.random.default_rng(seed)
    imp = importance(n)
    params = [rng.normal(size=(m, n)) * 0.3, np.zeros(n)]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        X = sample_features(rng, batch, n, sparsity)
        _, grad_W, grad_b = loss_and_grads(params[0], params[1], X, imp)
        for p, g, v, u in zip(params, (grad_W, grad_b), first, second, strict=True):
            v *= beta1
            v += (1 - beta1) * g
            u *= beta2
            u += (1 - beta2) * g * g
            p -= lr * (v / (1 - beta1**t)) / (np.sqrt(u / (1 - beta2**t)) + eps)
    return params[0], params[1]


def features_represented(W: Array) -> int:
    """
    Count the columns of ``W`` with norm above 0.5.

    A feature the model stores has a column of norm near 1; a dropped one has
    a column near 0.

    Examples
    --------
    >>> import numpy as np
    >>> features_represented(np.array([[1.0, 0.0, 0.6], [0.0, 0.1, 0.0]]))
    2
    """
    return int(np.sum(np.linalg.norm(W, axis=0) > 0.5))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEEING IT: ASCII Heatmap and Bar Chart (stdlib)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAMP: Final[str] = " .:-=+*#%@"


def shade(M: Array) -> str:
    """
    Draw ``M`` as text, one doubled character per entry, darker for larger ``|value|``.

    Entries are clipped to [0, 1] after taking the absolute value, then mapped
    onto ``RAMP``. For Wᵀ W, a bright diagonal marks stored features and bright
    off-diagonal cells mark the interference between them.

    Examples
    --------
    >>> import numpy as np
    >>> shade(np.array([[0.0, 1.0], [-0.5, 0.05]])).splitlines()
    ['  @@', '++  ']
    """
    rows = []
    for row in np.abs(M):
        cells = (RAMP[min(int(v * len(RAMP)), len(RAMP) - 1)] for v in np.minimum(row, 1.0))
        rows.append("".join(c * 2 for c in cells))
    return "\n".join(rows)


def bars(rows: Sequence[tuple[str, int]], width: int = 40, top: int | None = None) -> str:
    """
    Draw one bar per row, with lengths proportional to each value.

    ``top`` is the value that fills ``width`` characters; it defaults to the
    largest value. Labels are left-aligned to the longest one.

    Examples
    --------
    >>> print(bars([("a", 5), ("bb", 10)], width=10))
    a  |#####       5
    bb |##########  10
    >>> print(bars([("a", 5)], width=10, top=25))
    a |##          5
    """
    label_width = max(len(label) for label, _ in rows)
    full = top if top is not None else max(value for _, value in rows)
    lines = []
    for label, value in rows:
        length = round(value / full * width) if full > 0 else 0
        lines.append(f"{label:<{label_width}} |{'#' * length:<{width}}  {value}")
    return "\n".join(lines)


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
      LESSON INDEX: tier_5_lesson_2a_superposition
    ...
    Prerequisites (hard deps):
       → tier_5_lesson_1_manual_backprop: Hand-derived gradients
       → tier_4_lesson_1_finite_differences: Numeric gradients to check against
    ...
       → Power-up in: tier_5_lesson_3_pytorch
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


def demo(
    sparsities: Sequence[float] = (0.0, 0.7, 0.9, 0.97, 0.99),
    n: int = 20,
    m: int = 5,
    steps: int = 10_000,
    lr: float = 1e-3,
) -> None:
    """
    Sweep sparsity, then print the features represented and Wᵀ W for the first and last run.

    Examples
    --------
    >>> demo(sparsities=(0.0, 0.95), n=5, m=2, steps=2000, lr=0.01)  # doctest: +ELLIPSIS
    ═...
    sparsity    features represented
    0           2
    0.95        5
    ...
    """
    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    print()
    runs = [train(n, m, s, steps=steps, lr=lr) for s in sparsities]
    counts = [features_represented(W) for W, _ in runs]
    print(f"{'sparsity':<12}features represented")
    for s, count in zip(sparsities, counts, strict=True):
        print(f"{s:<12g}{count}")
    print()
    print(f"Features represented, out of {n} (hidden dimensions: {m}):")
    print(bars([(f"s={s:g}", c) for s, c in zip(sparsities, counts, strict=True)], top=n))
    for label, (W, _) in (("dense", runs[0]), ("sparsest", runs[-1])):
        s = sparsities[0] if label == "dense" else sparsities[-1]
        print()
        print(f"Wᵀ W at sparsity {s:g} ({label}); diagonal = stored, off-diagonal = interference:")
        print(shade(W.T @ W))


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
