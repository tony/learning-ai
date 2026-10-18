#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = []
# ///
"""
Lesson 4.2a: Why Momentum Works (side quest).

You run plain gradient descent and heavy-ball momentum on the same badly
conditioned bowl and count the steps each needs. Gradient descent pays for the
bowl's condition number κ (its steepest curvature divided by its flattest);
momentum pays only for √κ. At κ = 10,000 that is the difference between tens of
thousands of steps and about a thousand.

Source: Gabriel Goh, "Why Momentum Really Works", Distill (2017),
https://distill.pub/2017/momentum/

Prerequisites (must complete first)
----------------------------------
- Lesson 4.2: Gradient Descent + Learning Rate Failure Modes (step size, divergence)

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. Explain why one step size cannot suit a bowl's steep and flat directions at once.
2. Add a velocity term to gradient descent and tune its step size and decay.
3. Measure the √κ speedup and show that switching momentum off destroys it.

The Big Idea
------------
Momentum remembers the last step, so the flat directions keep moving while the
steep ones stop oscillating, and the cost falls from κ to √κ.

Why It Matters
--------------
Every optimizer you will train networks with (SGD with momentum, Adam, AdamW)
carries a velocity term. Poorly scaled features and deep networks produce exactly
the badly conditioned bowls where that term decides whether training takes
minutes or days.

Stdlib Pain → Library Power-Up
------------------------------
Pain (this lesson):
- Each coordinate is updated in a Python loop, one eigenvalue at a time.
- Lines of code: ~15 for the update loop.
- Runtime: about a second for the whole κ sweep up to 10,000.
- Common bugs: tuning the step size for gradient descent and reusing it with
  momentum, which diverges.

Power-Up (Lesson 5.3 — PyTorch):
- ``torch.optim.SGD(params, lr=alpha, momentum=beta)`` runs the same update on
  tensors of any shape.
- Key APIs: ``torch.optim.SGD``, ``optimizer.step()``.

Complexity Analysis
-------------------
Pain (manual), for n eigenvalues and k steps:
- Time:  O(n·k); k grows like κ for gradient descent and like √κ for momentum.
- Space: O(n).

Execution
---------
Project-style:
    uv run python src/042a_momentum.py

Script-style (self-contained deps):
    uv run --script src/042a_momentum.py

Options:
    python src/042a_momentum.py --doctest-only
    python src/042a_momentum.py --no-doctest
    python src/042a_momentum.py --index
    python src/042a_momentum.py -v

Quality Gates
-------------
    uv run pytest src/042a_momentum.py
    uv run ruff check src/042a_momentum.py
    uv run ruff format --check src/042a_momentum.py
    uv run mypy --strict src/042a_momentum.py

Module Doctests
---------------
>>> lam = curvatures(100.0, n=2)
>>> steps_to_converge(lam, *gd_params(lam))
691
>>> steps_to_converge(lam, *momentum_params(lam)) < 691 // 5
True
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON_ID: Final[str] = "tier_4_lesson_2a_momentum"

# Momentum's library form arrives with PyTorch's optimizers.
POWER_UP_AVAILABLE: Final[bool] = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason

PREREQUISITES: Final[Mapping[str, str]] = {
    "tier_4_lesson_2_gradient_descent": "Step size and the divergence it can cause",
}

DEPENDENCIES: Final[Mapping[str, str]] = {
    "tier_4_lesson_1_finite_differences": "What a gradient is, numerically",
}

SUCCESSORS: Final[Mapping[str, str]] = {
    "tier_4_lesson_3_linear_regression": "A real loss surface whose conditioning you can feel",
}

DEPENDENTS: Final[Mapping[str, str]] = {
    "tier_5_lesson_3_pytorch": "torch.optim.SGD(momentum=...) runs this update on tensors",
}

RELATED: Final[Mapping[str, str]] = {
    "tier_5_lesson_2a_superposition": "Trains with Adam, momentum plus per-coordinate scaling",
}

POWER_UP_IN_LESSON: Final[str | None] = "tier_5_lesson_3_pytorch"
PAIN_FROM_LESSON: Final[str | None] = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE PAIN: Gradient Descent and Momentum, One Coordinate at a Time
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def curvatures(kappa: float, n: int = 50) -> list[float]:
    """
    Return ``n`` curvatures spread evenly on a log scale from 1 to ``kappa``.

    The bowl f(w) = ½ Σ λᵢ wᵢ² has these λᵢ as its eigenvalues, so its condition
    number is exactly ``kappa``. ``n`` must be at least 2.

    Examples
    --------
    >>> curvatures(100.0, n=3)
    [1.0, 10.0, 100.0]
    >>> curvatures(9.0, n=2)
    [1.0, 9.0]
    """
    return [kappa ** (i / (n - 1)) for i in range(n)]


def gd_params(lam: Sequence[float]) -> tuple[float, float]:
    """
    Return the best fixed step size for plain gradient descent, and zero decay.

    The step 2 / (λmin + λmax) makes the flattest and steepest directions shrink
    by the same factor, (κ - 1) / (κ + 1), on every step.

    Examples
    --------
    >>> gd_params([1.0, 3.0])
    (0.5, 0.0)
    """
    return 2.0 / (min(lam) + max(lam)), 0.0


def momentum_params(lam: Sequence[float]) -> tuple[float, float]:
    """
    Return the best step size and decay for heavy-ball momentum.

    Step = (2 / (√λmin + √λmax))², decay = ((√λmax - √λmin) / (√λmax + √λmin))².
    Every direction then shrinks by √decay = (√κ - 1) / (√κ + 1) per step.

    Examples
    --------
    >>> alpha, beta = momentum_params([1.0, 100.0])
    >>> round(alpha, 6), round(beta, 6)
    (0.033058, 0.669421)
    """
    lo, hi = math.sqrt(min(lam)), math.sqrt(max(lam))
    return (2.0 / (lo + hi)) ** 2, ((hi - lo) / (hi + lo)) ** 2


def steps_to_converge(
    lam: Sequence[float],
    alpha: float,
    beta: float,
    tol: float = 1e-6,
    limit: int = 1_000_000,
) -> int:
    """
    Count the steps until every coordinate is within ``tol`` of the minimum.

    Starts from w = 1 in every coordinate with zero velocity and applies
    z ← beta·z + λ·w, then w ← w - alpha·z. ``beta = 0`` is plain gradient
    descent. Returns ``limit`` when the run diverges or stalls.

    Examples
    --------
    One step lands exactly on the minimum when alpha = 1 / λ:

    >>> steps_to_converge([2.0], alpha=0.5, beta=0.0)
    1

    The deliberate break: keep momentum's step size but switch its decay off.
    The steep direction overshoots and diverges, so the run hits the limit:

    >>> lam = [1.0, 100.0]
    >>> alpha, beta = momentum_params(lam)
    >>> steps_to_converge(lam, alpha, beta) < 120
    True
    >>> steps_to_converge(lam, alpha, beta=0.0, limit=10_000)
    10000
    """
    w = [1.0] * len(lam)
    z = [0.0] * len(lam)
    for step in range(1, limit + 1):
        for i, curvature in enumerate(lam):
            z[i] = beta * z[i] + curvature * w[i]
            w[i] -= alpha * z[i]
        worst = max(abs(x) for x in w)
        if worst < tol:
            return step
        if not math.isfinite(worst):
            return limit
    return limit


def sweep(kappas: Sequence[float], n: int = 50) -> list[tuple[float, int, int]]:
    """
    Return ``(kappa, gradient-descent steps, momentum steps)`` for each condition number.

    Each method runs with its own best parameters from ``gd_params`` and
    ``momentum_params``.

    Examples
    --------
    >>> [(k, gd) for k, gd, _ in sweep([10.0, 100.0], n=2)]
    [(10.0, 69), (100.0, 691)]
    """
    rows = []
    for kappa in kappas:
        lam = curvatures(kappa, n)
        gd = steps_to_converge(lam, *gd_params(lam))
        mom = steps_to_converge(lam, *momentum_params(lam))
        rows.append((kappa, gd, mom))
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEEING IT: A Log-Scale ASCII Bar Chart (stdlib)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def log_bars(rows: Sequence[tuple[str, int]], width: int = 40) -> str:
    """
    Draw one bar per row, with lengths proportional to log10 of each value.

    The largest value fills ``width`` characters; a value of 1 draws no bar.
    Values must be at least 1. Labels are left-aligned to the longest one.

    Examples
    --------
    >>> print(log_bars([("ten", 10), ("thousand", 1000)], width=6))
    ten      |##      10
    thousand |######  1000
    """
    label_width = max(len(label) for label, _ in rows)
    top = math.log10(max(value for _, value in rows))
    lines = []
    for label, value in rows:
        length = round(math.log10(value) / top * width) if top > 0 else 0
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
      LESSON INDEX: tier_4_lesson_2a_momentum
    ...
    Prerequisites (hard deps):
       → tier_4_lesson_2_gradient_descent: Step size and the divergence it can cause
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


def demo(kappas: Sequence[float] = (10.0, 100.0, 1_000.0, 10_000.0)) -> None:
    """
    Print the step counts for each condition number, as a table and as bars.

    Examples
    --------
    >>> demo(kappas=(10.0,))  # doctest: +ELLIPSIS
    ═...
    kappa       gradient descent    momentum    speedup
    10          69                  27          2.6x
    ...
    """
    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    print()
    rows = sweep(kappas)
    print(f"{'kappa':<12}{'gradient descent':<20}{'momentum':<12}speedup")
    for kappa, gd, mom in rows:
        print(f"{kappa:<12g}{gd:<20}{mom:<12}{gd / mom:.1f}x")
    print()
    print("Steps to converge (log scale):")
    bars = [
        (f"{name} k={kappa:g}", steps)
        for kappa, gd, mom in rows
        for name, steps in (("GD ", gd), ("mom", mom))
    ]
    print(log_bars(bars))


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
