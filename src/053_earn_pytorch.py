#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = ["numpy>=2.5.4", "torch>=2.14.1"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""
Lesson 5.3: Earn PyTorch — Tensor Autograd and a Training Loop Skeleton.

You derive the gradient of logistic regression by hand in NumPy, then let
PyTorch's autograd compute the same gradient from the forward pass alone, and
check that the two agree to rounding error. A broken derivation fails the same
check. Then you train the model twice, once with your gradient and once with
autograd and ``torch.optim``, and watch the two loss curves coincide.

Prerequisites (must complete first)
----------------------------------
- Lesson 5.2: Build Scalar Autodiff Engine (what ``.backward()`` automates)

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. Mark tensors with ``requires_grad=True``, call ``.backward()``, and read ``.grad``.
2. Check autograd against a hand-derived gradient, and see the check fail on a
   broken derivation.
3. Write a training loop (forward, backward, optimizer step, evaluate) that you
   can reuse for any model.

The Big Idea
------------
Autograd records the operations of the forward pass and replays them backwards,
so the gradient you derived by hand comes for free for any model you can write
as tensor operations.

Why It Matters
--------------
Every network after this lesson has too many parameters to differentiate by
hand. Trusting autograd is safe only because you have seen it match a
derivation you could check yourself.

Stdlib Pain → Library Power-Up
------------------------------
Pain (NumPy, hand-derived):
- The gradient of the mean binary cross-entropy, ∂L/∂w = Xᵀ(sigmoid(Xw + b) - y)/n,
  written out by hand.
- Lines of code: ~4 for this model, and a new derivation for every new model.
- Common bugs: a missing 1/n, a transposed X, a sign flip.

Power-Up (this lesson — PyTorch):
- ``loss.backward()`` fills ``w.grad`` and ``b.grad`` from the forward pass alone.
- ``torch.optim.SGD`` applies the update.
- Key APIs: ``torch.tensor(..., requires_grad=True)``, ``Tensor.backward``,
  ``torch.nn.functional.binary_cross_entropy_with_logits``, ``torch.optim.SGD``.

Dependencies
------------
PyPI carries no torch wheel for Python 3.15, which CI runs, so torch is not a
project dependency. The PEP 723 header above installs it from PyTorch's CPU
index; run this file with ``uv run --script``. ``src/conftest.py`` keeps pytest
from importing it where torch is absent.

Complexity Analysis
-------------------
For n samples and d features, one loss-and-gradient evaluation costs O(n·d)
time and O(n + d) extra memory, by hand or by autograd. Autograd also keeps the
forward pass's intermediate tensors until ``.backward()`` runs.

Execution
---------
Script-style (installs torch from PyTorch's CPU index):
    uv run --script src/053_earn_pytorch.py

Options:
    uv run --script src/053_earn_pytorch.py --doctest-only
    uv run --script src/053_earn_pytorch.py --no-doctest
    uv run --script src/053_earn_pytorch.py --index
    uv run --script src/053_earn_pytorch.py --device cuda

Quality Gates
-------------
    uv run --script src/053_earn_pytorch.py --doctest-only
    uv run ruff check src/053_earn_pytorch.py
    uv run ruff format --check src/053_earn_pytorch.py
    uv run mypy src/053_earn_pytorch.py

Module Doctests
---------------
>>> X, y = make_blobs(n=40, seed=0)
>>> check_parity(loss_and_grad_numpy, X, y) < 1e-12
True
"""

from __future__ import annotations

import argparse
import doctest
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

    from numpy.typing import NDArray

    type Array = NDArray[np.float64]
    type LossGrad = tuple[float, Array, float]
    type LossGradFn = Callable[[Array, float, Array, Array], LossGrad]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON_ID: Final[str] = "tier_5_lesson_3_pytorch"

POWER_UP_AVAILABLE: Final[bool] = True

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason

PREREQUISITES: Final[Mapping[str, str]] = {
    "tier_5_lesson_2_autodiff_engine": "A scalar autodiff engine: what .backward() automates",
}

DEPENDENCIES: Final[Mapping[str, str]] = {
    "tier_5_lesson_2a_superposition": "Hand-derived gradients checked against finite differences",
    "tier_2_lesson_4_cross_entropy": "The loss being minimized",
}

SUCCESSORS: Final[Mapping[str, str]] = {
    "tier_5_lesson_4_shape_discipline": "Tensor shapes as checked invariants",
}

DEPENDENTS: Final[Mapping[str, str]] = {
    "tier_8_lesson_2_tiny_gpt": "Trains a transformer with this loop",
}

RELATED: Final[Mapping[str, str]] = {
    "tier_4_lesson_2a_momentum": "torch.optim.SGD(momentum=...) runs that lesson's update",
}

POWER_UP_IN_LESSON: Final[str | None] = None
PAIN_FROM_LESSON: Final[str | None] = "tier_5_lesson_1_manual_backprop"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA: Two Gaussian Blobs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def make_blobs(n: int = 200, seed: int = 0) -> tuple[Array, Array]:
    """
    Return ``n`` 2-D points in two Gaussian blobs and their 0/1 labels.

    The first half centre on (-1.5, -1.5) with label 0, the rest on (1.5, 1.5)
    with label 1, each with unit spread, so a line separates them almost
    perfectly.

    Examples
    --------
    >>> X, y = make_blobs(n=6, seed=0)
    >>> X.shape, y.tolist()
    ((6, 2), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    """
    rng = np.random.default_rng(seed)
    y = np.repeat([0.0, 1.0], [n // 2, n - n // 2])
    centres = np.where(y[:, None] == 1.0, 1.5, -1.5)
    return centres + rng.standard_normal((n, 2)), y


def init_params(d: int, seed: int = 0) -> tuple[Array, float]:
    """
    Return small random starting weights for ``d`` features, and a zero bias.

    Both trainers start from these, so their runs can be compared step by step.

    Examples
    --------
    >>> w, b = init_params(2, seed=0)
    >>> w.shape, b
    ((2,), 0.0)
    """
    return 0.01 * np.random.default_rng(seed).standard_normal(d), 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE PAIN: The Gradient, Derived by Hand
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def sigmoid(z: Array) -> Array:
    """
    Return sigmoid(z) = 1 / (1 + e⁻ᶻ), without overflow for large |z|.

    Examples
    --------
    >>> sigmoid(np.array([0.0, 800.0, -800.0])).tolist()
    [0.5, 1.0, 0.0]
    """
    return np.asarray(np.exp(-np.logaddexp(0.0, -z)), dtype=np.float64)


def loss_and_grad_numpy(w: Array, b: float, X: Array, y: Array) -> LossGrad:
    """
    Return the mean binary cross-entropy of logistic regression and its gradient.

    With logits z = Xw + b the loss is mean(log(1 + eᶻ) - y·z). Its derivative
    with respect to each zᵢ is (sigmoid(zᵢ) - yᵢ)/n, so
    ∂L/∂w = Xᵀ(sigmoid(z) - y)/n and ∂L/∂b = mean(sigmoid(z) - y).

    Examples
    --------
    At w = 0 and b = 0 every prediction is ½, so the loss is ln 2, and balanced
    labels leave the bias gradient at zero:

    >>> X, y = make_blobs(n=4, seed=0)
    >>> loss, grad_w, grad_b = loss_and_grad_numpy(np.zeros(2), 0.0, X, y)
    >>> round(loss, 6), round(grad_b, 6)
    (0.693147, 0.0)
    """
    z = X @ w + b
    loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
    residual = sigmoid(z) - y
    return loss, X.T @ residual / len(y), float(np.mean(residual))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE POWER-UP: Autograd
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def derivative(f: Callable[[torch.Tensor], torch.Tensor], x: float) -> float:
    """
    Return df/dx at ``x`` by autograd, for ``f`` written with tensor operations.

    Examples
    --------
    f(x) = x³ + 2x has f'(x) = 3x² + 2, so f'(2) = 14:

    >>> derivative(lambda t: t**3 + 2 * t, 2.0)
    14.0
    """
    t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    f(t).backward()
    return float(t.grad.item())


def loss_and_grad_torch(w: Array, b: float, X: Array, y: Array, device: str = "cpu") -> LossGrad:
    """
    Return the same loss written as tensor operations, with the gradient from autograd.

    Nothing here mentions sigmoid(z) - y: ``loss.backward()`` works it out from
    the forward pass.

    Examples
    --------
    >>> X, y = make_blobs(n=4, seed=0)
    >>> loss, grad_w, grad_b = loss_and_grad_torch(np.zeros(2), 0.0, X, y)
    >>> round(loss, 6), round(grad_b, 6)
    (0.693147, 0.0)
    """
    w_t = torch.tensor(w, dtype=torch.float64, device=device, requires_grad=True)
    b_t = torch.tensor(b, dtype=torch.float64, device=device, requires_grad=True)
    x_t = torch.as_tensor(X, dtype=torch.float64, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float64, device=device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(x_t @ w_t + b_t, y_t)
    loss.backward()
    grad_w = np.asarray(w_t.grad.cpu().numpy(), dtype=np.float64)
    return float(loss.item()), grad_w, float(b_t.grad.item())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PARITY: Does the Hand Derivation Match Autograd?
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def check_parity(
    loss_and_grad: LossGradFn, X: Array, y: Array, trials: int = 5, seed: int = 1
) -> float:
    """
    Return the largest gap between ``loss_and_grad`` and autograd at random parameters.

    Compares the loss and both gradients at ``trials`` random (w, b) points. A
    correct derivation agrees to rounding error, around 1e-16.

    Examples
    --------
    >>> X, y = make_blobs(n=40, seed=0)
    >>> check_parity(loss_and_grad_numpy, X, y) < 1e-12
    True

    The deliberate break: forget the 1/n in the weight gradient. The check
    catches it:

    >>> def forgot_mean(w, b, X, y):
    ...     loss, grad_w, grad_b = loss_and_grad_numpy(w, b, X, y)
    ...     return loss, grad_w * len(y), grad_b
    >>> check_parity(forgot_mean, X, y) < 1e-12
    False
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        w, b = rng.standard_normal(int(X.shape[1])), float(rng.standard_normal())
        loss, grad_w, grad_b = loss_and_grad(w, b, X, y)
        ref_loss, ref_grad_w, ref_grad_b = loss_and_grad_torch(w, b, X, y)
        gaps = (abs(loss - ref_loss), float(np.max(np.abs(grad_w - ref_grad_w))))
        worst = max(worst, *gaps, abs(grad_b - ref_grad_b))
    return worst


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING: The Same Loop, by Hand and with torch.optim
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class TrainResult:
    """One training run's history.

    Attributes
    ----------
    losses : list of float
        The training loss before each update, one per epoch.
    accuracy : float
        Fraction of training points the final model labels correctly.
    """

    losses: list[float]
    accuracy: float


def train_numpy(
    X: Array, y: Array, w: Array, b: float, epochs: int = 100, lr: float = 0.5
) -> TrainResult:
    """
    Fit logistic regression with the hand-derived gradient and plain gradient descent.

    Examples
    --------
    >>> X, y = make_blobs(n=200, seed=0)
    >>> result = train_numpy(X, y, *init_params(2), epochs=100)
    >>> result.losses[0] > result.losses[-1], result.accuracy > 0.95
    (True, True)
    """
    losses = []
    for _ in range(epochs):
        loss, grad_w, grad_b = loss_and_grad_numpy(w, b, X, y)
        w, b = w - lr * grad_w, b - lr * grad_b
        losses.append(loss)
    accuracy = float(np.mean((X @ w + b > 0) == (y == 1.0)))
    return TrainResult(losses, accuracy)


def train_torch(
    X: Array,
    y: Array,
    w: Array,
    b: float,
    epochs: int = 100,
    lr: float = 0.5,
    device: str = "cpu",
) -> TrainResult:
    """
    Fit the same model with autograd and ``torch.optim.SGD``: the reusable loop.

    Every epoch runs the four steps any PyTorch model trains with: clear the old
    gradients, run the forward pass, call ``.backward()``, step the optimizer.

    Examples
    --------
    From the same start, the two trainers produce the same loss curve:

    >>> X, y = make_blobs(n=200, seed=0)
    >>> w, b = init_params(2)
    >>> by_hand = train_numpy(X, y, w, b, epochs=50)
    >>> by_torch = train_torch(X, y, w, b, epochs=50)
    >>> max(abs(p - q) for p, q in zip(by_hand.losses, by_torch.losses, strict=True)) < 1e-12
    True
    >>> by_torch.accuracy == by_hand.accuracy
    True
    """
    x_t = torch.as_tensor(X, dtype=torch.float64, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float64, device=device)
    w_t = torch.tensor(w, dtype=torch.float64, device=device, requires_grad=True)
    b_t = torch.tensor(b, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([w_t, b_t], lr=lr)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x_t @ w_t + b_t, y_t)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    with torch.no_grad():
        correct = (x_t @ w_t + b_t > 0) == (y_t == 1.0)
        accuracy = float(correct.double().mean().item())
    return TrainResult(losses, accuracy)


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
    device : str
        Where ``train_torch`` runs: ``cpu`` by default, or ``cuda`` on a GPU.
    epochs : int
        Training epochs for the demo.
    """

    doctest: bool
    doctest_only: bool
    verbose: bool
    show_index: bool
    device: str
    epochs: int


def parse_args(argv: list[str] | None = None) -> Args:
    """
    Parse the lesson's command-line flags.

    Examples
    --------
    >>> parse_args(["--doctest-only"]).doctest_only
    True
    >>> parse_args([]).device
    'cpu'
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
    parser.add_argument("--device", default="cpu", help="Device for train_torch (default: cpu)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    ns = parser.parse_args(argv)
    return Args(
        doctest=ns.doctest,
        doctest_only=ns.doctest_only,
        verbose=ns.verbose,
        show_index=ns.show_index,
        device=ns.device,
        epochs=ns.epochs,
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
      LESSON INDEX: tier_5_lesson_3_pytorch
    ...
       ← Pain from: tier_5_lesson_1_manual_backprop
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


def demo(device: str = "cpu", epochs: int = 100) -> None:
    """
    Print autograd's derivative, the parity check, and the two loss curves.

    Examples
    --------
    >>> demo(epochs=20)  # doctest: +ELLIPSIS
    ═...
    d/dx (x³ + 2x) at x = 2: autograd 14.0, by hand 14.0
    parity, hand-derived vs autograd: largest gap ...
    <BLANKLINE>
    epoch    by hand     torch.optim
    0        0.693162    0.693162
    ...
    training accuracy: by hand 0.980, torch.optim 0.980
    """
    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    print()
    print(
        f"d/dx (x³ + 2x) at x = 2: autograd {derivative(lambda t: t**3 + 2 * t, 2.0)}, by hand 14.0"
    )
    X, y = make_blobs(n=200, seed=0)
    gap = check_parity(loss_and_grad_numpy, X, y)
    print(f"parity, hand-derived vs autograd: largest gap {gap:.1e}")
    print()
    w, b = init_params(X.shape[1])
    by_hand = train_numpy(X, y, w, b, epochs=epochs)
    by_torch = train_torch(X, y, w, b, epochs=epochs, device=device)
    print(f"{'epoch':<9}{'by hand':<12}torch.optim")
    for epoch in range(0, epochs, max(1, epochs // 5)):
        print(f"{epoch:<9}{by_hand.losses[epoch]:<12.6f}{by_torch.losses[epoch]:.6f}")
    print(f"training accuracy: by hand {by_hand.accuracy:.3f}, torch.optim {by_torch.accuracy:.3f}")


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
    demo(device=args.device, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
